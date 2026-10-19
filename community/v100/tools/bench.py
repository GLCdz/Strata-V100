#!/usr/bin/env python3
"""Measure a running server with synthetic prompts; output statistics only, no response text."""
import argparse
import hashlib
import json
import pathlib
import re
import statistics
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
TASKS = {
    'code': '阅读以下 Python 函数并修复其中的差一错误与未关闭的资源句柄，给出修正后的完整代码：\n\ndef load_rows(path):\n    f = open(path)\n    rows = f.read().splitlines()\n    for i in range(len(rows)):\n        print(rows[i+1])\n',
    'zh': '写一段 300 字左右的中文技术说明，解释注意力机制中的 KV 缓存为什么随上下文线性增长，以及稀疏注意力如何缓解这个问题。',
    'en': 'Explain in about 200 words why mixture-of-experts models can scale total parameters while keeping per-token compute low, and name one systems challenge this creates.',
}

def build_prompt(tokens):
    text = re.sub(r'\s+', ' ', (ROOT / 'corpora/strata-paper.txt').read_text())
    chars = int(tokens / 0.3083)
    parts = []
    length = 0
    while length < chars:
        part = f'[块{len(parts)+1}] ' + text
        parts.append(part)
        length += len(part)
    return ' '.join(parts)[:chars]

def associate(before, after, usage, started):
    old = {json.dumps(r, sort_keys=True) for r in before['requests']}
    new = [r for r in after['requests'] if json.dumps(r, sort_keys=True) not in old]
    matches = [r for r in new if r.get('prompt_tokens') == usage['prompt_tokens']
               and r.get('output_tokens') == usage['completion_tokens'] and r.get('time', 0) >= started]
    if len(new) != 1 or len(matches) != 1:
        raise RuntimeError('Ambiguous metrics association or another client was active; discard this run')
    return matches[0]

def stream(opener, request):
    first = None
    usage = None
    parts = []
    started = time.time()
    with opener.open(request, timeout=900) as response:
        for line in response:
            if not line.startswith(b'data: ') or b'[DONE]' in line:
                continue
            data = json.loads(line[6:])
            if data.get('error'):
                raise RuntimeError('Server returned a streaming error')
            if data.get('usage'):
                usage = data['usage']
            for choice in data.get('choices', []):
                content = choice.get('delta', {}).get('content')
                if content:
                    if first is None:
                        first = time.time() - started
                    parts.append(content)
    if not usage or not usage.get('completion_tokens') or first is None:
        raise RuntimeError('Missing nonempty visible output or token usage')
    return started, first, usage, hashlib.sha256(''.join(parts).encode()).hexdigest()

class Runner:
    def __init__(self, config, url=None):
        self.config = config
        self.url = (url or f"http://127.0.0.1:{config.get('port', 8080)}").rstrip('/')
        self.headers = {'Authorization': 'Bearer ' + config['api_key'], 'Content-Type': 'application/json'}
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def metrics(self):
        req = urllib.request.Request(self.url + '/metrics', headers=self.headers)
        with self.opener.open(req, timeout=8) as response:
            return json.load(response)

    def ask(self, case, text, output=768, fresh=False):
        before = self.metrics()
        live = before['live']
        if live['state'] != 'idle' or any(live.get(k, 0) for k in ('queued', 'running', 'waiting', 'admitting')):
            raise RuntimeError('Other requests are active; benchmark requires exclusive idle service')
        body = {'model': self.config['model_name'], 'messages': [{'role': 'user', 'content': text}],
                'stream': True, 'temperature': 0, 'max_tokens': output,
                'chat_template_kwargs': {'enable_thinking': False}}
        req = urllib.request.Request(self.url + '/v1/chat/completions', json.dumps(body).encode(), self.headers)
        started, ttft, usage, digest = stream(self.opener, req)
        e = associate(before, self.metrics(), usage, started)
        if fresh and e['reused'] != 0:
            raise RuntimeError('Fresh prompt reused history; discard this measurement')
        return {'case': case, 'prompt_tokens': e['prompt_tokens'], 'reused_tokens': e['reused'],
                'prompt_read_tokens': e['prompt_read'], 'output_tokens': e['output_tokens'],
                'prompt_ms': e['prompt_ms'], 'decode_ms': e['decode_ms'], 'decode_tps': e['decode_tok_s'],
                'prefill_tps': e['prompt_read'] * 1000 / e['prompt_ms'] if e['prompt_ms'] else None,
                'ttft_s': ttft, 'output_sha256': digest}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=pathlib.Path, required=True)
    ap.add_argument('--url')
    ap.add_argument('--output', type=pathlib.Path, required=True)
    ap.add_argument('--sizes', nargs='+', type=int, default=[4096,16384,32768,130000,240000])
    ap.add_argument('--repeats', type=int, default=2)
    ap.add_argument('--decode-repeats', type=int, default=3)
    a = ap.parse_args()
    if a.repeats < 1 or a.decode_repeats < 1 or any(n < 1 or n > 240000 for n in a.sizes):
        ap.error('Use positive repeat counts and prompt targets in 1..240000')
    runner = Runner(json.loads(a.config.read_text()), a.url)
    rows = []
    # Reserve the output path before sending any model requests; never replace an old run.
    with a.output.open('x') as f:
        runner.ask('warmup', '用一句话解释什么是熔断器模式。', 48)
        for key,text in TASKS.items():
            for _ in range(a.decode_repeats):
                row = runner.ask('decode-'+key, text)
                rows.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
        for size in a.sizes:
            for _ in range(a.repeats):
                text = f'新鲜度标记 {time.time_ns() % 1000000000}，读完文档后回答：文档的主题是什么？\n' + build_prompt(size)
                row = runner.ask('fresh-en-'+str(size), text, 24, fresh=True)
                rows.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
        background = build_prompt(32000) + '\n\n以上是背景文档。现在忽略文档内容，直接回答：\n' + TASKS['code']
        for _ in range(a.repeats):
            row = runner.ask('decode-code-32k', background)
            rows.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
    for case in dict.fromkeys(r['case'] for r in rows):
        values = [r['prefill_tps'] if case.startswith('fresh-') else r['decode_tps'] for r in rows if r['case']==case]
        print(f'{case}: n={len(values)}, median={statistics.median(values):.1f} t/s')

if __name__ == '__main__':
    main()
