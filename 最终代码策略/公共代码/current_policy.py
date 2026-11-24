"""唯一当前策略入口。读取发布选择；官方连接仍必须人工确认演练。"""
from pathlib import Path
import argparse,json,hashlib,sys
from solve import PracticeHTTP,FormalHTTP

ROOT=Path(__file__).resolve().parents[1]


def configuration():
    return json.loads((ROOT/'配置/当前策略.json').read_text(encoding='utf-8'))


def final_source_hashes():
    """记录本次实际加载的项目源码，避免把历史实验脚本混入证据。"""
    hashes={}
    for module in tuple(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if not filename:continue
        path=Path(filename).resolve()
        if path.suffix!='.py' or not path.is_relative_to(ROOT):continue
        relative=path.relative_to(ROOT).as_posix()
        hashes[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(hashes.items()))


def make_robot(backend,question,**kwargs):
    spec=configuration()['questions'][str(question)]
    if question==3 and spec['engine']=='ccr_js':
        from q3_ccr_js import CCRJSQ3
        return CCRJSQ3(backend,question,stop_on_discovery=spec['stop_on_discovery'],**kwargs)
    if question==4 and spec['engine']=='vc_svs':
        from q4_vc_svs import VCSVQ4
        return VCSVQ4(backend,question,variant=spec['variant'],**kwargs)
    raise RuntimeError(f'问题{question}的最终引擎配置不一致，拒绝猜测运行')


def main():
    parser=argparse.ArgumentParser(description='连接人工启动并确认的演练或正式会话；程序自身不启动测试')
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--practice-question',type=int,choices=[3,4])
    mode.add_argument('--formal-question',type=int,choices=[3,4])
    parser.add_argument('--robot-id',required=True)
    parser.add_argument('--practice-guard');parser.add_argument('--formal-guard')
    parser.add_argument('--output',required=True);args=parser.parse_args()
    out=Path(args.output);log=out.with_suffix('.jsonl')
    if out.exists() or log.exists():raise RuntimeError('输出已存在，禁止覆盖已有测试记录')
    out.parent.mkdir(parents=True,exist_ok=True)
    formal=args.formal_question is not None
    question=args.formal_question if formal else args.practice_question
    config=configuration()
    if formal:
        if not config.get('official_tests_allowed'):raise RuntimeError('当前配置未授权正式测试')
        if not args.formal_guard or args.practice_guard:raise RuntimeError('正式模式必须且只能提供formal-guard')
        client=FormalHTTP(args.robot_id,args.formal_guard,question)
        evidence='official_formal_test'
    else:
        if not args.practice_guard or args.formal_guard:raise RuntimeError('演练模式必须且只能提供practice-guard')
        client=PracticeHTTP(args.robot_id,args.practice_guard,question)
        evidence='official_practice_not_formal'
    robot=make_robot(client,question,log_path=log)
    result=robot.run();result.update(case_code=client.guard['case_code'],
        states={str(k):b.status for k,b in robot.states.items()},optical_plans=getattr(robot,'optical_plans',[]),
        evidence_type=evidence,test_mode='formal' if formal else 'practice',policy_selection=config,
        source_sha256=final_source_hashes())
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if result.get('status')!='completed':raise SystemExit(2)


if __name__=='__main__':main()
