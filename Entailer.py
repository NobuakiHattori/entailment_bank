#まずはgit cloneして、Entailment Bankのデータをダウンロードしておくこと

# Load the model （11bの方）
from utils.nlp_agent import MultiAngleModel, NlpAgent
ew_model = MultiAngleModel(model_path="allenai/entailer-11b", cuda_devices=0)
prover = NlpAgent(model=ew_model, default_outputs="proof")
entail_verifier = NlpAgent(model=ew_model, default_outputs=["implied"], default_options={"explicit_outputs": ['true', 'false']})
hyp_verifier = NlpAgent(model=ew_model, default_outputs=["valid"], default_options={"explicit_outputs": ['true', 'false']})

def prove_recursively(hypothesis, depth=0, max_depth=3, proven=None): #4スコア導入再帰証明関数
    if proven is None:
        proven = {}

    indent = "  " * depth
    print(f"{indent}🧠 Trying to prove: \"{hypothesis}\"")

    # ① 直接スコア（仮説そのものの妥当性）
    validity = hyp_verifier({"hypothesis": hypothesis})
    raw_score = validity.get("output_prob", 0.0)
    direct_score = raw_score if validity["valid"] == "true" else 1.0 - raw_score
    print(f"{indent}   ↳ Direct Score: {direct_score:} → {'✅' if validity['valid']=='true' else '❌'}")

    # ② 前提を生成（必ず生成）
    proof = prover({"hypothesis": hypothesis})
    premises = [x.strip() for x in proof.split("[PREMISE]") if x.strip()]
    if premises:
        print(f"{indent}   ↳ 🔍 Generated Premises:")
        for p in premises:
            print(f"{indent}     - {p}")
    else:
        print(f"{indent}   ↳ ❌ No premises generated.")

    # ③ 含意スコア
    entaility = entail_verifier({"hypothesis": hypothesis, "proof": proof})
    raw_e_score = entaility.get("output_prob", 0.0)
    entail_score = raw_e_score if entaility["implied"] == "true" else 1.0 - raw_e_score
    print(f"{indent}   ↳ Entailment Score: {entail_score:}→ {'✅' if entaility['implied']=='true' else '❌'}")

    # ④ 証明スコア（前提がある場合に評価）
    if premises and depth + 1 < max_depth:
        premise_scores = []
        for premise in premises:
            premise_validity = hyp_verifier({"hypothesis": premise})
            raw_score = premise_validity.get("output_prob", 0.0)
            adjusted_score = raw_score if premise_validity["valid"] == "true" else 1.0 - raw_score
            premise_scores.append(adjusted_score)
        premise_product = 1.0
        for s in premise_scores:
            premise_product *= s
        proof_score = entail_score * premise_product
        print(f"{indent}   ↳ Proof Score = {entail_score:} × Π(Premises) = {proof_score:}")
    else:
        proof_score = 0.0
        if not premises:
            print(f"{indent}   ↳ No premises → Proof Score: 0.0")
        else:
            print(f"{indent}   ↳ Max depth reached → Skipping premise evaluation")

    # ⑤ 総合スコア（大きい方を採用）
    if direct_score >= proof_score:
        total_score = direct_score
        print(f"{indent}   ✅ Total Score = Direct ({direct_score}) ≥ Proof ({proof_score}) → Stop here.")
        proven[hypothesis] = total_score
        return total_score
    else:
        total_score = proof_score
        print(f"{indent}   🔁 Total Score = Proof ({proof_score}) > Direct ({direct_score}) → Recurse on premises.")

    # ⑥ 採用された場合のみ再帰
    if depth + 1 < max_depth:
        for premise in premises:
            prove_recursively(premise, depth + 1, max_depth, proven)
    else:
        print(f"{indent}   ⛔ Reached max depth ({max_depth})")

    proven[hypothesis] = total_score
    return total_score

