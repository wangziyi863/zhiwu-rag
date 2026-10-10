"""Small localhost-only UI; heavy jobs are serialized across requests."""
import base64
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import secrets
import tempfile
import threading
from urllib.parse import unquote, urlparse


def serve(pipeline, port):
    token = secrets.token_urlsafe(32)
    lock = threading.Lock()
    page = (Path(__file__).parent/'web.html').read_text().replace('__TOKEN__', token)

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body, content_type='application/json; charset=utf-8'):
            data = body if isinstance(body, bytes) else (json.dumps(body, ensure_ascii=False).encode() if not isinstance(body,str) else body.encode())
            self.send_response(status)
            self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(data)))
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Cache-Control','no-store')
            self.end_headers()
            self.wfile.write(data)

        def valid_host(self):
            return self.headers.get('Host') in {f'127.0.0.1:{port}', f'localhost:{port}'}

        def do_GET(self):
            if not self.valid_host():
                self.send(403,{'error':'Localhost access only'})
                return
            path = unquote(urlparse(self.path).path)
            if path == '/':
                self.send(200,page,'text/html; charset=utf-8')
            elif path == '/api/status':
                self.send(200,dict(pipeline.status(),busy=lock.locked()))
            elif path.startswith('/assets/'):
                target = (pipeline.jobs/path[len('/assets/'):]).resolve()
                allowed = target.is_relative_to(pipeline.jobs) and target.suffix in ('.png','.jpg')
                if allowed and target.is_file():
                    self.send(200,target.read_bytes(),mimetypes.guess_type(target.name)[0])
                else:
                    self.send(404,{'error':'Not found'})
            else:
                self.send(404,{'error':'Not found'})

        def do_POST(self):
            if not self.valid_host() or self.headers.get('X-RAG-Token') != token:
                self.send(403,{'error':'请从本机网页发起请求。'})
                return
            if not lock.acquire(blocking=False):
                self.send(409,{'error':'GPU 正在处理另一项任务，请稍后重试。'})
                return
            try:
                length = int(self.headers.get('Content-Length','0'))
                if not 0 < length <= 20_000_000:
                    raise ValueError('请求为空或过大，图片请控制在约 12 MB 内。')
                data = json.loads(self.rfile.read(length))
                if self.path == '/api/index':
                    result = pipeline.build_index(data['catalog'])
                elif self.path == '/api/ask':
                    raw = base64.b64decode(data['image'].split(',',1)[-1],validate=True)
                    with tempfile.NamedTemporaryFile(suffix='.upload',dir=pipeline.home) as upload:
                        upload.write(raw)
                        upload.flush()
                        result = pipeline.ask(upload.name,data['question'],data.get('compression','location'),
                                              data.get('retrieval_mode','region'),bool(data.get('retrieval_only',False)),
                                              bool(data.get('repair_enabled',True)),
                                              bool(data.get('dual_recall',True)),bool(data.get('detailed_categories',False)),
                                              data.get('recall_k',12))
                else:
                    self.send(404,{'error':'Not found'})
                    return
                self.send(200,result)
            except Exception as error:
                self.send(400,{'error':str(error)})
            finally:
                lock.release()

    server = ThreadingHTTPServer(('127.0.0.1',port),Handler)
    print(f'区域 RAG 已启动：http://localhost:{port} （Ctrl+C 停止）',flush=True)
    server.serve_forever()
