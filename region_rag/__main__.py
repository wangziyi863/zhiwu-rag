import argparse
import json

from .pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(description='GLA-CLIP 自动区域检索与问答')
    parser.add_argument('--home', help='索引和查询记录目录')
    sub = parser.add_subparsers(dest='command', required=True)
    index = sub.add_parser('index', help='导入 JSONL/CSV 知识库')
    index.add_argument('--catalog', required=True)
    ask = sub.add_parser('ask', help='自动选区、检索并回答')
    ask.add_argument('--image', required=True)
    ask.add_argument('--question', required=True)
    ask.add_argument('--compression', choices=['none','location','sim08'],default='location')
    ask.add_argument('--retrieval-mode', choices=['region','whole'],default='region')
    ask.add_argument('--retrieval-only',action='store_true')
    ask.add_argument('--no-repair',action='store_true',help='关闭自动粘连修复，运行基线对照')
    ask.add_argument('--no-dual-recall',action='store_true')
    ask.add_argument('--detailed-categories',action='store_true',help='实验性细化类别描述，默认关闭')
    ask.add_argument('--recall-k',type=int,default=12)
    web = sub.add_parser('serve', help='打开本地网页服务')
    web.add_argument('--port',type=int,default=7860)
    sub.add_parser('status')
    args = parser.parse_args()
    pipeline = Pipeline(args.home)
    if args.command == 'index':
        result = pipeline.build_index(args.catalog)
    elif args.command == 'ask':
        result = pipeline.ask(args.image,args.question,args.compression,args.retrieval_mode,args.retrieval_only,not args.no_repair,
                              not args.no_dual_recall,args.detailed_categories,args.recall_k)
    elif args.command == 'serve':
        from .web import serve
        serve(pipeline, args.port)
        return
    else:
        result = pipeline.status()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
