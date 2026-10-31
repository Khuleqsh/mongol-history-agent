import argparse
import json
import sys

from .config import EVAL_DIR, GAPS, SOURCES


def _print_answer(r: dict) -> None:
    print(r["answer"])
    if r["sources"]:
        print("\n--- Эх сурвалж ---")
        for label, s in r["sources"].items():
            sec = f', {s["section"]}' if s.get("section") else ""
            print(f'[{label}] {s["title"]} — {s["author"]} {s["year"]} ({s["lang"]}){sec}, chunk {s["chunk"]}')
    if r["invalid_citations"]:
        print("\n⚠ Олдоогүй ишлэл:", ", ".join(r["invalid_citations"]))
    if r["unsupported_removed"]:
        print(f'\n(Давхар шалгалтаар {len(r["unsupported_removed"])} батлагдаагүй мэдэгдлийг засав)')


def main(argv=None) -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    p = argparse.ArgumentParser(prog="mha", description="Монголын түүхийн судлаач агент")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check-sources", help="sources.json-ийг шалгах")
    sub.add_parser("ingest", help="индекс бүтээх")

    levels = ["standard", "brief", "student", "researcher"]
    a = sub.add_parser("ask", help="нэг асуулт асуух")
    a.add_argument("question")
    a.add_argument("--level", choices=levels, default="standard")
    a.add_argument("--verify", action="store_true", help="ишлэлийг давхар шалгах (удаан)")
    c = sub.add_parser("chat", help="олон дахин ярилцах")
    c.add_argument("--level", choices=levels, default="standard")
    c.add_argument("--verify", action="store_true")

    f = sub.add_parser("feedback", help="хариултад үнэлгээ өгөх")
    f.add_argument("question")
    f.add_argument("verdict", choices=["good", "bad"])
    f.add_argument("--note", default="")

    sub.add_parser("gaps", help="архивт дутсан сэдвүүдийн жагсаалт")
    e = sub.add_parser("eval", help="eval багц ажиллуулах")
    e.add_argument("--update-baseline", action="store_true")
    args = p.parse_args(argv)

    if args.cmd == "check-sources":
        from .ingest import validate_sources

        errs = validate_sources(json.loads(SOURCES.read_text(encoding="utf-8")))
        print("OK" if not errs else "\n".join(errs))
        sys.exit(1 if errs else 0)
    elif args.cmd == "ingest":
        from .ingest import build_index

        build_index()
    elif args.cmd == "ask":
        from .agent import Agent

        _print_answer(Agent().ask(args.question, level=args.level, verify=args.verify))
    elif args.cmd == "chat":
        from .agent import Agent

        agent, history = Agent(), []
        print("Гарах: Ctrl+C. Асуултаа Монгол эсвэл English-ээр бич.")
        while True:
            try:
                q = input("\n> ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not q:
                continue
            r = agent.ask(q, level=args.level, history=history, verify=args.verify)
            _print_answer(r)
            history += [{"role": "user", "content": q}, {"role": "assistant", "content": r["answer"]}]
    elif args.cmd == "feedback":
        from .feedback import log_feedback

        log_feedback(args.question, "", args.verdict, args.note)
    elif args.cmd == "gaps":
        from .gaps import summarize_gaps

        rows = summarize_gaps(GAPS)
        if not rows:
            print("Бүртгэгдсэн цоорхой алга.")
        for topic, count, example in rows:
            print(f"{count:>3} × {topic}   (жишээ: {example})")
    elif args.cmd == "eval":
        from . import evalkit
        from .agent import Agent

        agent = Agent()
        items = evalkit.load_questions(EVAL_DIR / "questions.jsonl")
        score = evalkit.run_eval(agent, items)
        print(f"Оноо: {score:.2%}")
        evalkit.append_history(EVAL_DIR / "history.jsonl", score, agent.prompt_name, agent.model, len(items))
        baseline_path = EVAL_DIR / "baseline.json"
        if args.update_baseline:
            baseline_path.write_text(json.dumps({"score": score}))
        elif evalkit.regressed(score, evalkit.read_baseline(baseline_path)):
            print("Baseline-аас буурсан.")
            sys.exit(1)


if __name__ == "__main__":
    main()
