"""Real-model smoke test for the Assembler.

Seeds a realistic textbook-page worth of Elements into a throwaway Neo4j, then runs
the Assembler with a live LLM (DeepSeek via litellm) and prints the Blocks it built.

Run:  python scripts/smoke_assembler.py
Needs DEEPSEEK_API_KEY in the environment (or in the file named by SMOKE_ENV_FILE).
No secrets are printed.
"""
import asyncio
import os

from dotenv import dotenv_values
from testcontainers.neo4j import Neo4jContainer

from math_trainer.core.config import DatabaseConfig, StageConfig
from math_trainer.core.dspy.module import DSPyModule
from math_trainer.core.model.types import EdgeType, NodeType
from math_trainer.ingestion.stages.assembler import AssemblerStage
from math_trainer.ingestion.stages.signatures import AssemblerSignature
from math_trainer.storage.neo4j.driver import Neo4jDriver
from math_trainer.storage.neo4j.repository import GraphRepository

# Env vars win; fall back to an optional dotenv file (SMOKE_ENV_FILE).
CREDS = {**dotenv_values(os.environ.get("SMOKE_ENV_FILE", "")), **os.environ}

# A realistic §-of-a-textbook page, already extracted into typed Elements.
# (uuid, type, content, blurb)
ELEMENTS = [
    ("e1", NodeType.HEADING, "## 3.2 The Derivative", None),
    ("e2", NodeType.PARAGRAPH, "We now make precise the idea of an instantaneous rate of change.", None),
    ("e3", NodeType.PARAGRAPH, "Definition 3.1. The derivative of $f$ at $a$ is the limit of the difference quotient, when it exists.", None),
    ("e4", NodeType.MATH, "$$f'(a) = \\lim_{h \\to 0} \\frac{f(a+h) - f(a)}{h}.$$", None),
    ("e5", NodeType.PARAGRAPH, "Theorem 3.4 (Power Rule). For any integer $n$, if $f(x)=x^n$ then $f'(x)=nx^{n-1}$.", None),
    ("e6", NodeType.PARAGRAPH, "Proof. Expand $(x+h)^n$ by the binomial theorem and cancel the leading term.", None),
    ("e7", NodeType.PARAGRAPH, "Example 3.6. Differentiate $g(x)=x^3$.", None),
    ("e8", NodeType.MATH, "$$g'(x) = 3x^{2}.$$", None),
    ("e9", NodeType.IMAGE, None, "A tangent line touching the curve y=x^3 at a point."),
    ("e10", NodeType.INSTRUCTION, "Exercises 1-3. Differentiate each function.", None),
    ("e11", NodeType.ACTIVITY, "1. $f(x) = x^{5}$", None),
    ("e12", NodeType.ACTIVITY, "2. $f(x) = \\sin x$", None),
    ("e13", NodeType.ACTIVITY, "3. $f(x) = e^{x}$", None),
]
INSTRUCTS = [("e10", "e11"), ("e10", "e12"), ("e10", "e13")]


async def seed(repo: GraphRepository, source_uuid: str) -> None:
    await repo.create_node([NodeType.SOURCE], uuid=source_uuid, title="Calculus §3.2", stage=6)
    for i, (uid, ntype, content, blurb) in enumerate(ELEMENTS, start=1):
        await repo.create_node(
            [NodeType.ELEMENT, ntype], uuid=uid, source_uuid=source_uuid,
            order_index=i, content=content, blurb=blurb, page_no=1,
        )
    for ins, act in INSTRUCTS:
        await repo.link(EdgeType.INSTRUCTS, ins, act)


async def main() -> None:
    with Neo4jContainer("neo4j:5.26-community") as container:
        db = DatabaseConfig(uri=container.get_connection_url(), user="neo4j",
                            password=container.password, database="neo4j")
        driver = Neo4jDriver(db)
        repo = GraphRepository(driver)
        source_uuid = "smoke-src-1"
        await seed(repo, source_uuid)

        stage_cfg = StageConfig(
            model="deepseek/deepseek-chat",
            api_key=CREDS.get("DEEPSEEK_API_KEY", ""),
            main_window_tokens=1200, context_window_tokens=300,
            timeout_s=120.0, num_retries=2,
        )
        module = DSPyModule(stage_cfg, AssemblerSignature)
        assembler = AssemblerStage(repo, module, stage_cfg)

        print("Running Assembler against deepseek/deepseek-chat …\n")
        await assembler.run(source_uuid)

        blocks = await repo.source_blocks_ordered(source_uuid)
        print(f"=== {len(blocks)} blocks built ===\n")
        for b in blocks:
            members = await repo.run(
                "MATCH (:`Block` {uuid:$u})-[:`Groups`]->(e) "
                "RETURN e.uuid AS uuid ORDER BY e.order_index", u=b["uuid"],
            )
            member_ids = [m["uuid"] for m in members]
            label = f" [{b['label']}]" if b.get("label") else ""
            preview = (b.get("content") or "").replace("\n", " ")
            preview = preview[:90] + ("…" if len(preview) > 90 else "")
            print(f"• {b['kind']:<11}{label} members={member_ids}")
            print(f"    {preview}\n")

        await driver.close()


if __name__ == "__main__":
    asyncio.run(main())
