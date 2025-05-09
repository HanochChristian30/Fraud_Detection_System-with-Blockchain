"""
Bayesian Risk Scoring Utilities for Fraud Detection
==================================================

This module provides utilities for applying Bayesian risk scoring to
both in-person and online credit card transactions.
"""

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime

class BayesianRiskCalculator:
    """
    Calculates Bayesian risk scores for credit card transactions.
    Can be used for both in-person and online transactions.
    """
    
    def __init__(self, prior_fraud_prob=0.01):
        """
        Initialize the risk calculator with the prior probability of fraud.
        
        Args:
            prior_fraud_prob (float): The base rate of fraud before considering any evidence.
                                     Should be between 0 and 1.
        """
        self.prior_fraud_prob = prior_fraud_prob
        
        # Define high-risk categories
        self.high_risk_categories = [
            'electronics', 'jewelry', 'gift_card', 'gambling',
            'digital_goods', 'cryptocurrency', 'money_transfer'
        ]
        
        # Define medium-risk categories
        self.medium_risk_categories = [
            'clothing', 'travel', 'entertainment', 'dining', 
            'furniture', 'automotive', 'subscription'
        ]
        
        # Other categories are considered low risk
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate distance between two geographic points using the Haversine formula.
        
        Args:
            lat1, lon1: Latitude and longitude of first point
            lat2, lon2: Latitude and longitude of second point
            
        Returns:
            float: Distance in kilometers
        """
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        radius = 6371  # Radius of the Earth in kilometers
        
        return radius * c
    
    def calculate_inperson_risk(self, transaction_data):
        """
        Calculate Bayesian risk for in-person transaction.
        
        Args:
            transaction_data (dict): Dictionary with transaction details
            
        Returns:
            dict: Risk analysis results with probabilities and likelihood ratios
        """
        # Extract transaction data
        hour = transaction_data.get('hour', datetime.now().hour)
        amount = transaction_data.get('amount', 0)
        category = transaction_data.get('category', '').lower()
        lat = transaction_data.get('lat')
        long = transaction_data.get('long')
        merch_lat = transaction_data.get('merch_lat')
        merch_long = transaction_data.get('merch_long')
        birth_year = transaction_data.get('birth_year')
        gender = transaction_data.get('gender')
        merchant = transaction_data.get('merchant', '')
        
        # Calculate likelihood ratios
        lr_time = self._time_risk(hour)
        lr_amount = self._amount_risk(amount)
        lr_category = self._category_risk(category)
        lr_location = self._location_risk(lat, long, merch_lat, merch_long)
        lr_age = self._age_risk(birth_year)
        lr_merchant = self._merchant_risk(merchant)
        lr_gender = self._gender_risk(gender)
        
        # Calculate combined likelihood ratio
        likelihoods = {
            'time': lr_time,
            'amount': lr_amount,
            'category': lr_category,
            'location': lr_location,
            'age': lr_age,
            'merchant': lr_merchant,
            'gender': lr_gender
        }
        
        # Remove any None values
        likelihoods = {k: v for k, v in likelihoods.items() if v is not None}
        
        # Calculate combined likelihood ratio
        combined_lr = 1.0
        for lr in likelihoods.values():
            combined_lr *= lr
        
        # Apply Bayes' theorem
        result = self._apply_bayes(combined_lr, likelihoods)
        
        return result
    
    def calculate_online_risk(self, transaction_data):
        """
        Calculate Bayesian risk for online transaction.
        
        Args:
            transaction_data (dict): Dictionary with online transaction details
            
        Returns:
            dict: Risk analysis results with probabilities and likelihood ratios
        """
        # Extract transaction data
        hour = transaction_data.get('hour', datetime.now().hour)
        amount = transaction_data.get('amount', 0)
        category = transaction_data.get('category', '').lower()
        birth_year = transaction_data.get('birth_year')
        ip_address = transaction_data.get('ip_address')
        device_id = transaction_data.get('device_id')
        is_new_device = transaction_data.get('is_new_device', True)
        account_age_days = transaction_data.get('account_age_days', 0)
        is_digital_goods = transaction_data.get('is_digital_goods', False)
        shipping_address = transaction_data.get('shipping_address')
        billing_address = transaction_data.get('billing_address')
        ip_country = transaction_data.get('ip_country')
        billing_country = transaction_data.get('billing_country')
        transaction_velocity = transaction_data.get('transaction_velocity', 0)
        email_domain = transaction_data.get('email_domain', '')
        
        # Calculate likelihood ratios
        lr_time = self._time_risk(hour)
        lr_amount = self._amount_risk(amount)
        lr_category = self._category_risk(category)
        lr_age = self._age_risk(birth_year)
        lr_device = self._device_risk(is_new_device)
        lr_account = self._account_age_risk(account_age_days)
        lr_digital = self._digital_goods_risk(is_digital_goods)
        lr_address = self._address_mismatch_risk(shipping_address, billing_address)
        lr_ip_location = self._ip_location_risk(ip_country, billing_country)
        lr_velocity = self._velocity_risk(transaction_velocity)
        lr_email = self._email_risk(email_domain)
        
        # Calculate combined likelihood ratio
        likelihoods = {
            'time': lr_time,
            'amount': lr_amount,
            'category': lr_category,
            'age': lr_age,
            'device': lr_device,
            'account_age': lr_account,
            'digital_goods': lr_digital,
            'address_mismatch': lr_address,
            'ip_location': lr_ip_location,
            'velocity': lr_velocity,
            'email': lr_email
        }
        
        # Remove any None values
        likelihoods = {k: v for k, v in likelihoods.items() if v is not None}
        
        # Calculate combined likelihood ratio
        combined_lr = 1.0
        for lr in likelihoods.values():
            combined_lr *= lr
        
        # Apply Bayes' theorem
        result = self._apply_bayes(combined_lr, likelihoods)
        
        return result
    
    def _apply_bayes(self, combined_lr, likelihoods):
        """
        Apply Bayes' theorem to calculate posterior probability.
        
        Args:
            combined_lr (float): Combined likelihood ratio
            likelihoods (dict): Individual likelihood ratios
            
        Returns:
            dict: Results with probabilities and risk factors
        """
        # Calculate posterior odds using Bayes' formula
        prior_odds = self.prior_fraud_prob / (1 - self.prior_fraud_prob)
        posterior_odds = prior_odds * combined_lr
        
        # Convert odds to probability
        posterior_prob = posterior_odds / (1 + posterior_odds)
        
        return {
            'prior_probability': self.prior_fraud_prob,
            'posterior_probability': posterior_prob,
            'likelihood_ratio': combined_lr,
            'individual_likelihoods': likelihoods
        }
    
    def _time_risk(self, hour):
        """Calculate risk factor based on time of day"""
        if hour is None:
            return 1.2  # Slightly elevated risk if time unknown
        
        if hour < 6 or hour > 22:
            return 3.0  # Late night/early morning 3x more likely to be fraud
        elif hour < 9 or hour > 18:
            return 1.5  # Outside business hours somewhat more risky
        else:
            return 1.0  # Normal business hours baseline risk
    
    def _amount_risk(self, amount):
        """Calculate risk factor based on transaction amount"""
        if amount is None:
            return 1.2  # Slightly elevated risk if amount unknown
        
        if amount > 1000:
            return 5.0  # Large transactions 5x more likely to be fraud
        elif amount > 500:
            return 2.0  # Medium transactions 2x more likely
        elif amount > 200:
            return 1.2  # Slightly above average
        else:
            return 1.0  # Small transactions baseline risk
    
    def _category_risk(self, category):
        """Calculate risk factor based on merchant category"""
        if not category:
            return 1.2  # Slightly elevated risk if category unknown
        
        if category in [c.lower() for c in self.high_risk_categories]:
            return 4.0  # High risk categories 4x more likely to be fraud
        elif category in [c.lower() for c in self.medium_risk_categories]:
            return 2.0  # Medium risk categories 2x more likely
        else:
            return 1.0  # Low risk categories baseline risk
    
    def _location_risk(self, lat, long, merch_lat, merch_long):
        """Calculate risk factor based on distance between customer and merchant"""
        if None in (lat, long, merch_lat, merch_long):
            return 1.5  # Slightly elevated risk when location unknown
        
        try:
            distance = self.calculate_distance(lat, long, merch_lat, merch_long)
            
            if distance > 100:
                return 8.0  # Very far distances 8x more likely to be fraud
            elif distance > 50:
                return 4.0  # Far distances 4x more likely
            elif distance > 20:
                return 2.0  # Moderate distances 2x more likely
            else:
                return 1.0  # Close distances baseline risk
        except:
            return 1.5  # Error calculating distance
    
    def _age_risk(self, birth_year):
        """Calculate risk factor based on customer age"""
        if not birth_year:
            return 1.2  # Slightly elevated risk when age unknown
        
        try:
            current_year = datetime.now().year
            age = current_year - int(birth_year)
            
            if age < 25:
                return 1.5  # Younger people slightly higher risk
            elif age > 70:
                return 1.3  # Elderly slightly higher risk (targets of scams)
            else:
                return 1.0  # Middle-aged baseline risk
        except:
            return 1.2  # Error calculating age
    
    def _merchant_risk(self, merchant):
        """Calculate risk factor based on merchant reputation"""
        # In a real system, this would reference a database of merchant risk scores
        # For now, we'll just return a placeholder value
        return 1.0  # Default merchant risk
    
    def _gender_risk(self, gender):
        """Calculate risk factor based on gender"""
        # This might be controversial, but if your data shows gender correlations,
        # you could implement it here
        return None  # Not using gender as a risk factor
    
    def _device_risk(self, is_new_device):
        """Calculate risk factor based on device newness"""
        if is_new_device is None:
            return 1.2  # Slightly elevated risk if device info unknown
        
        return 6.0 if is_new_device else 1.0
    
    def _account_age_risk(self, account_age_days):
        """Calculate risk factor based on account age"""
        if account_age_days is None:
            return 1.3  # Slightly elevated risk if account age unknown
        
        if account_age_days < 7:
            return 5.0  # Very new accounts 5x more likely to be fraud
        elif account_age_days < 30:
            return 2.0  # New accounts 2x more likely
        elif account_age_days < 90:
            return 1.3  # Somewhat new accounts slightly more risky
        else:
            return 1.0  # Established accounts baseline risk
    
    def _digital_goods_risk(self, is_digital_goods):
        """Calculate risk factor for digital vs physical goods"""
        if is_digital_goods is None:
            return 1.0  # No change if unknown
        
        return 2.0 if is_digital_goods else 1.0
    
    def _address_mismatch_risk(self, shipping_address, billing_address):
        """Calculate risk factor based on shipping/billing address match"""
        # In a real system, you'd do proper address comparison
        if None in (shipping_address, billing_address):
            return 1.2  # Slightly elevated risk if address info incomplete
        
        return 3.0 if shipping_address != billing_address else 1.0
    
    def _ip_location_risk(self, ip_country, billing_country):
        """Calculate risk factor based on IP geolocation vs billing country"""
        if None in (ip_country, billing_country):
            return 1.5  # Elevated risk if location info incomplete
        
        return 7.0 if ip_country != billing_country else 1.0
    
    def _velocity_risk(self, transaction_velocity):
        """Calculate risk factor based on transaction velocity"""
        if transaction_velocity is None:
            return 1.0  # No change if unknown
        
        if transaction_velocity > 10:
            return 6.0  # Very high velocity 6x more likely to be fraud
        elif transaction_velocity > 5:
            return 4.0  # High velocity 4x more likely
        elif transaction_velocity > 3:
            return 2.0  # Moderate velocity 2x more likely
        else:
            return 1.0  # Low velocity baseline risk
    
    def _email_risk(self, email_domain):
        """Calculate risk factor based on email domain reputation"""
        if not email_domain:
            return 1.3  # Slightly elevated risk if email unknown
        
        # High risk domains (in a real system, this would be more comprehensive)
        high_risk_domains = ['tempmail.com', 'guerrillamail.com', 'mailinator.com']
        
        # Check for disposable email
        if email_domain.lower() in high_risk_domains:
            return 8.0  # Disposable email 8x more likely to be fraud
        
        # Check for free email
        free_email_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']
        if email_domain.lower() in free_email_domains:
            return 1.5  # Free email slightly more likely to be fraud vs. business email
        
        return 1.0  # Business email baseline risk