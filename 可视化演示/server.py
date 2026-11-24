"""仅监听本机的离线演示：合成场景、策略运行与保存日志回放。"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import importlib.util
import json
import mimetypes
import sys
import threading
import traceback
import urllib.parse
import tempfile

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / '最终代码策略'))
import 入口 as final_policy
sys.path.append(str(ROOT / 'src'))
from benchmark import LocalSimulator, scenario

lock = threading.RLock()
state = dict(state='idle', events=[], sources=[], error=None, question=None, seed=None)


def recordings():
    rows=[]
    for group in ('本地合成','正式测试','官方演练'):
        directory=ROOT/'输出'/group
        for p in sorted(directory.glob('Q*.jsonl')):
            rows.append(dict(path=p.relative_to(ROOT).as_posix(),label=f'{group} · {p.stem}'))
    return rows


def replay(path):
    p=(ROOT/path).resolve()
    if not p.is_relative_to((ROOT/'输出').resolve()) or p.suffix.lower()!='.jsonl' or not p.is_file():
        raise ValueError('只能回放项目输出目录内的 JSONL 记录')
    events=[]
    for line in p.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip(): continue
        e=json.loads(line)
        if e.get('path') not in ('/enter','/measure','/clear','/exit'):continue
        response=e.get('response') or {}
        events.append(dict(path=e['path'],position=e.get('position'),channel=e.get('channel'),
                           response=response,virtual_time_s=response.get('virtual_time_s',0)))
    if not events:raise ValueError('记录中没有可播放的动作')
    with lock:
        if state['state']=='running':raise ValueError('本地场景仍在运行')
        state.update(state='done',events=events,sources=[],error=None,question=None,seed=None)


def run_job(question,seed,strategy_path,strategy_code=None,strategy_name=None):
    temp=None
    try:
        sources,meta=scenario(question,seed)
        simulator=LocalSimulator(sources,seed,meta['error_mode'])
        with lock:state['sources']=[dict(channel=s.channel,x=s.x,y=s.y,radius=s.radius,direction=s.direction) for s in sources]

        class PublicBackend:
            def post(self,path,position=None,channel=None):
                response=simulator.post(path,position,channel)
                event=dict(path=path,position=list(map(float,position)) if position is not None else None,
                           channel=channel,response=response,virtual_time_s=response.get('virtual_time_s',0))
                with lock:state['events'].append(event)
                return response

        if strategy_code is not None:
            temp=tempfile.TemporaryDirectory(prefix='b_policy_')
            name=Path(strategy_name or 'strategy.py').name
            if not name.lower().endswith('.py'):name='strategy.py'
            path=Path(temp.name)/name
            path.write_text(strategy_code,encoding='utf-8')
        elif strategy_path:
            path=Path(strategy_path).expanduser().resolve()
        else:path=None
        if path is not None:
            if not path.is_file() or path.suffix.lower()!='.py':raise ValueError('策略路径必须指向本机 Python 文件')
            spec=importlib.util.spec_from_file_location('user_demo_strategy',path)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            factory=getattr(module,'make_robot',None)
            if not callable(factory):raise ValueError('策略文件需要提供 make_robot(backend, question)')
            robot=factory(PublicBackend(),question)
        else:robot=final_policy.make_robot(PublicBackend(),question)
        result=robot.run()
        with lock:state.update(state='done',result=result)
    except Exception:
        with lock:state.update(state='error',error=traceback.format_exc(limit=5))
    finally:
        if temp is not None:temp.cleanup()


class Handler(BaseHTTPRequestHandler):
    def send_json(self,data,status=200):
        raw=json.dumps(data,ensure_ascii=False,default=str).encode('utf-8')
        self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)

    def do_GET(self):
        path=urllib.parse.urlsplit(self.path).path
        if path=='/api/recordings':return self.send_json(dict(recordings=recordings()))
        if path=='/api/status':
            with lock:return self.send_json(state)
        name='index.html' if path=='/' else urllib.parse.unquote(path.lstrip('/'))
        p=(HERE/name).resolve()
        if not p.is_relative_to(HERE) or not p.is_file() or p.suffix not in ('.html','.js','.css','.svg'):
            self.send_error(404);return
        data=p.read_bytes();self.send_response(200)
        self.send_header('Content-Type',mimetypes.guess_type(str(p))[0] or 'application/octet-stream')
        self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)

    def do_POST(self):
        try:
            origin=self.headers.get('Origin')
            allowed=f'http://127.0.0.1:{self.server.server_port}'
            if self.headers.get('Host') != f'127.0.0.1:{self.server.server_port}':
                raise ValueError('仅允许通过本机 127.0.0.1 地址访问')
            if origin and origin != allowed:raise ValueError('仅允许本地页面发起请求')
            if not self.headers.get('Content-Type','').lower().startswith('application/json'):
                raise ValueError('请求必须使用 application/json')
            size=int(self.headers.get('Content-Length','0'))
            if size>262144:raise ValueError('请求过大，策略文件上限约256KB')
            data=json.loads(self.rfile.read(size) or b'{}')
            if self.path=='/api/replay':replay(data['path']);return self.send_json(dict(ok=True))
            if self.path=='/api/run':
                question=int(data['question']);seed=int(data.get('seed',324089))
                if question not in (3,4):raise ValueError('题号必须是3或4')
                code=data.get('strategy_code');path=data.get('strategy_path')
                if code is not None and not isinstance(code,str):raise ValueError('策略内容必须为文本')
                if code is not None and path:raise ValueError('策略内容与路径只能提供一个')
                if code is not None and not code.strip():raise ValueError('策略文件为空')
                with lock:
                    if state['state']=='running':raise ValueError('已有本地场景运行中')
                    state.clear();state.update(state='running',events=[],sources=[],error=None,question=question,seed=seed)
                threading.Thread(target=run_job,args=(question,seed,path,code,data.get('strategy_name')),daemon=True).start()
                return self.send_json(dict(ok=True))
            self.send_error(404)
        except (ValueError,KeyError,TypeError,json.JSONDecodeError) as exc:self.send_json(dict(error=str(exc)),400)


if __name__=='__main__':
    import argparse,webbrowser
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8765);parser.add_argument('--no-browser',action='store_true');args=parser.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'本地演示：http://127.0.0.1:{args.port}',flush=True)
    if not args.no_browser:webbrowser.open(f'http://127.0.0.1:{args.port}')
    server.serve_forever()
