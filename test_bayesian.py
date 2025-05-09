"""
Test script for Bayesian Risk Scoring module
==========================================

This script tests the BayesianRiskCalculator with various transaction scenarios
to ensure it works correctly for both in-person and online transactions.
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add the parent directory to the path to import bayesian_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bayesian_utils import BayesianRiskCalculator

def format_results(results):
    """Format the results for better printing"""
    lines = []
    lines.append(f"Prior Probability: {results['prior_probability']:.2%}")
    lines.append(f"Posterior Probability: {results['posterior_probability']:.2%}")
    lines.append(f"Combined Likelihood Ratio: {results['likelihood_ratio']:.2f}x")
    lines.append("\nIndividual Likelihood Ratios:")
    
    for factor, lr in results['individual_likelihoods'].items():
        lines.append(f"  {factor.replace('_', ' ').title()}: {lr:.2f}x")
    
    return "\n".join(lines)

def test_inperson_transactions():
    """Test Bayesian risk scoring for various in-person transaction scenarios"""
    print("\n=== TESTING IN-PERSON TRANSACTIONS ===\n")
    
    # Create the calculator
    calculator = BayesianRiskCalculator(prior_fraud_prob=0.01)
    
    # Test cases
    test_cases = [
        {
            "name": "Low Risk Transaction",
            "data": {
                "hour": 14,
                "amount": 50,
                "category": "grocery",
                "lat": 40.7128,
                "long": -74.0060,
                "merch_lat": 40.7130,
                "merch_long": -74.0065,
                "birth_year": 1980,
                "gender": "M",
                "merchant": "Local Grocery"
            }
        },
        {
            "name": "Medium Risk Transaction",
            "data": {
                "hour": 21,
                "amount": 700,
                "category": "electronics",
                "lat": 40.7128,
                "long": -74.0060,
                "merch_lat": 40.7500,
                "merch_long": -74.1000,
                "birth_year": 2000,
                "gender": "F",
                "merchant": "Big Electronics"
            }
        },
        {
            "name": "High Risk Transaction",
            "data": {
                "hour": 3,
                "amount": 2500,
                "category": "jewelry",
                "lat": 40.7128,
                "long": -74.0060,
                "merch_lat": 42.3601,
                "merch_long": -71.0589,  # Boston coordinates (far from NYC)
                "birth_year": 1950,
                "gender": "M",
                "merchant": "Luxury Jewels"
            }
        }
    ]
    
    # Run tests
    for test in test_cases:
        print(f"\n--- {test['name']} ---")
        results = calculator.calculate_inperson_risk(test["data"])
        print(format_results(results))
        print(f"FRAUD LIKELIHOOD: {'HIGH' if results['posterior_probability'] > 0.5 else 'MEDIUM' if results['posterior_probability'] > 0.1 else 'LOW'}")

def test_online_transactions():
    """Test Bayesian risk scoring for various online transaction scenarios"""
    print("\n=== TESTING ONLINE TRANSACTIONS ===\n")
    
    # Create the calculator with higher prior for online
    calculator = BayesianRiskCalculator(prior_fraud_prob=0.025)
    
    # Test cases
    test_cases = [
        {
            "name": "Low Risk Online Transaction",
            "data": {
                "hour": 14,
                "amount": 50,
                "category": "digital_goods",
                "birth_year": 1985,
                "is_new_device": False,
                "account_age_days": 365,
                "is_digital_goods": True,
                "shipping_address": "123 Main St, New York, NY",
                "billing_address": "123 Main St, New York, NY",
                "ip_country": "US",
                "billing_country": "US",
                "transaction_velocity": 1,
                "email_domain": "company.com"
            }
        },
        {
            "name": "Medium Risk Online Transaction",
            "data": {
                "hour": 22,
                "amount": 500,
                "category": "electronics",
                "birth_year": 1990,
                "is_new_device": True,
                "account_age_days": 45,
                "is_digital_goods": False,
                "shipping_address": "456 Oak St, Chicago, IL",
                "billing_address": "123 Main St, New York, NY",
                "ip_country": "US",
                "billing_country": "US",
                "transaction_velocity": 3,
                "email_domain": "gmail.com"
            }
        },
        {
            "name": "High Risk Online Transaction",
            "data": {
                "hour": 3,
                "amount": 1200,
                "category": "gift_card",
                "birth_year": 2000,
                "is_new_device": True,
                "account_age_days": 2,
                "is_digital_goods": True,
                "shipping_address": "Digital Delivery",
                "billing_address": "123 Main St, New York, NY",
                "ip_country": "RU",
                "billing_country": "US",
                "transaction_velocity": 12,
                "email_domain": "tempmail.com"
            }
        }
    ]
    
    # Run tests
    for test in test_cases:
        print(f"\n--- {test['name']} ---")
        results = calculator.calculate_online_risk(test["data"])
        print(format_results(results))
        print(f"FRAUD LIKELIHOOD: {'HIGH' if results['posterior_probability'] > 0.5 else 'MEDIUM' if results['posterior_probability'] > 0.1 else 'LOW'}")

def test_hybrid_approach():
    """Test the hybrid approach combining ML models with Bayesian analysis"""
    print("\n=== TESTING HYBRID APPROACH ===\n")
    
    # Create test cases with both ML probabilities and transaction data
    test_cases = [
        {
            "name": "ML and Bayesian Agree (Low Risk)",
            "ml_probability": 0.05,
            "bayesian_data": {
                "hour": 14,
                "amount": 50,
                "category": "grocery",
                "birth_year": 1980,
                "is_new_device": False,
                "account_age_days": 365,
                "is_digital_goods": False,
                "ip_country": "US",
                "billing_country": "US"
            },
            "ml_weight": 0.6,
            "bayesian_weight": 0.4
        },
        {
            "name": "ML and Bayesian Agree (High Risk)",
            "ml_probability": 0.85,
            "bayesian_data": {
                "hour": 3,
                "amount": 2500,
                "category": "gift_card",
                "birth_year": 2000,
                "is_new_device": True,
                "account_age_days": 5,
                "is_digital_goods": True,
                "ip_country": "RU",
                "billing_country": "US"
            },
            "ml_weight": 0.6,
            "bayesian_weight": 0.4
        },
        {
            "name": "ML and Bayesian Disagree (ML Low, Bayesian High)",
            "ml_probability": 0.1,
            "bayesian_data": {
                "hour": 3,
                "amount": 2000,
                "category": "jewelry",
                "birth_year": 1995,
                "is_new_device": True,
                "account_age_days": 2,
                "is_digital_goods": True,
                "ip_country": "CN",
                "billing_country": "US"
            },
            "ml_weight": 0.6,
            "bayesian_weight": 0.4
        },
        {
            "name": "ML and Bayesian Disagree (ML High, Bayesian Low)",
            "ml_probability": 0.75,
            "bayesian_data": {
                "hour": 14,
                "amount": 50,
                "category": "grocery",
                "birth_year": 1970,
                "is_new_device": False,
                "account_age_days": 1000,
                "is_digital_goods": False,
                "ip_country": "US",
                "billing_country": "US"
            },
            "ml_weight": 0.6,
            "bayesian_weight": 0.4
        }
    ]
    
    calculator = BayesianRiskCalculator(prior_fraud_prob=0.025)
    
    # Run tests
    for test in test_cases:
        print(f"\n--- {test['name']} ---")
        
        # Calculate Bayesian probability
        bayesian_result = calculator.calculate_online_risk(test["bayesian_data"])
        bayesian_probability = bayesian_result["posterior_probability"]
        
        # Get ML probability from test case
        ml_probability = test["ml_probability"]
        
        # Calculate hybrid probability
        hybrid_probability = (
            test["ml_weight"] * ml_probability + 
            test["bayesian_weight"] * bayesian_probability
        )
        
        # Print results
        print(f"ML Model Probability: {ml_probability:.2%}")
        print(f"Bayesian Probability: {bayesian_probability:.2%}")
        print(f"Hybrid Probability: {hybrid_probability:.2%}")
        
        # Determine risk level
        if hybrid_probability > 0.5:
            risk_level = "HIGH"
        elif hybrid_probability > 0.1:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        print(f"FRAUD LIKELIHOOD: {risk_level}")
        print("Bayesian Details:")
        print(f"  Prior Probability: {bayesian_result['prior_probability']:.2%}")
        print(f"  Likelihood Ratio: {bayesian_result['likelihood_ratio']:.2f}x")
        
        # Print top risk factors
        sorted_factors = sorted(
            bayesian_result['individual_likelihoods'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        print("Top Risk Factors:")
        for factor, value in sorted_factors[:3]:
            if value > 1.0:
                print(f"  {factor.replace('_', ' ').title()}: {value:.2f}x")

def main():
    """Run all tests"""
    print("\nBAYESIAN RISK SCORING TEST\n" + "=" * 28)
    
    # Test in-person transactions
    test_inperson_transactions()
    
    # Test online transactions
    test_online_transactions()
    
    # Test hybrid approach
    test_hybrid_approach()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()