Title: Validating Vulnerability Probes: Do Probes Beat Baselines?

Lexical code vulnerability detectors are commonly used in the industry to scan codebases. In this project, I was inspired by hallucination probes' success [\cite{hallucinationprobes}] to try detecting vulnerable code via model activations. The North Star of the project is to reduce risk in model sandbagging/sabotage with cheap but effective monitoring. The proxy task is to detect vulnerability awareness in existing code through model activations. The results are somewhat interesting, but not impressive. My conclusion is that there's still a lot of work to be done to narrow down if LLMs know they're writing vulnerable code or not, but LLMs should not be used to detect vulnerabilities in existing code.

## Dataset 

I started with the SVEN dataset. SVEN is a paired vulnerable-safe function pair dataset by He, Vechev. It contains pairs of vulnerable and patched functions with the type of vulnerability (CWE) marked. I processed it to get a token-level dataset by marking the tokens affected by the diff as positive in the vulnerable pair.

Later, I split by CWE, language and also isolated subtractive pairs. These were important to isolate effects of dataset imbalances and biases.

I later cross-checked with the PrimeVul dataset, which provides C/C++-only vulnerable pairs, which I could use in the exact same way.

