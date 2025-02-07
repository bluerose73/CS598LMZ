# CS598LMZ
Course project for CS598LMZ: Software Quality Assurance with Generative AI at UIUC

## Finalize topic / proposal
- MS & PhD (40-50)
- deadline: 2025-02-20
- project: 2025-04-29

## Code LLM
- Pre-training: (tons of GPUs)
- Post-training: backup.

## LLMs / Agents for Software QA
- 2.1 AI software engineer


## Some ideas:
- post-training: code diff? (sft?)
- pre-training: foundamental - solve it in pre-training
    - code suggestion: 
        - (1) updated code knowledge: python latest vs one version before; sdk old version vs new version.
            - code migration/upgrade.
        - (2) new code knowledge.  
    - model customization for each repo (RAG, sft, etc.)
        - RAG: code-embedding. (def add(a, b): return a + b vs def fun(x, y): return x + y)
        - Better Codebase Understanding: We've trained a new model for Codebase Understanding

## Strategy
- 1. literature review (AI software engineer: Devin, etc.): 4 days
- 2. literature review code post-training: 4 days
- 3. pre-training: supermaven/long context: 1 - 2 days
- 4. learning from spider / benchmarking: new code editing/diff benchmark (with new or updated knowledge): 1 - 2 days

