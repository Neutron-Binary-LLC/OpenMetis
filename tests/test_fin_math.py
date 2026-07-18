import torch
from nn.fin_math import FinMathTools

def test_fin_math():
    print("Testing FinMathTools...")
    
    # 1. list_available_models
    models = FinMathTools.list_available_models()
    print(f"Available models: {models}")
    assert "black_scholes" in models
    
    # 2. price_option
    params = {
        "S": torch.tensor([100.0]),
        "K": torch.tensor([100.0]),
        "T": torch.tensor([1.0]),
        "r": torch.tensor([0.05]),
        "sigma": torch.tensor([0.2])
    }
    price = FinMathTools.price_option("black_scholes", params)
    print(f"BS Price: {price.item():.4f}")
    assert price.item() > 0
    
    # 3. compute_greeks
    greeks = FinMathTools.compute_greeks("black_scholes", params)
    print(f"Greeks: { {k: v.item() for k, v in greeks.items()} }")
    assert "delta" in greeks
    assert "vega" in greeks
    
    # 4. monte_carlo_simulation
    mc_price = FinMathTools.monte_carlo_simulation("monte_carlo_bs", params, n_sim=100000)
    print(f"MC Price: {mc_price.item():.4f}")
    assert abs(mc_price.item() - price.item()) < 0.1
    
    # 5. calibrate_model
    market_data = {
        "price": price,
        "S": params["S"],
        "K": params["K"],
        "T": params["T"],
        "r": params["r"]
    }
    calib = FinMathTools.calibrate_model(market_data, "black_scholes")
    print(f"Calibrated Sigma: {calib['sigma'].item():.4f}")
    assert abs(calib['sigma'].item() - 0.2) < 1e-3
    
    # 6. risk_metrics
    risk = FinMathTools.risk_metrics({}, "black_scholes", params)
    print(f"Risk Metrics: {risk}")
    
    # 7. verify_consistency
    consistency = FinMathTools.verify_consistency([price, mc_price])
    print(f"Inconsistency (std): {consistency.item():.4f}")
    
    print("All tests passed!")

if __name__ == "__main__":
    test_fin_math()
