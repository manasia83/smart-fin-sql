# DS_fixed_income_training_generator.py

import pandas as pd
import random
from faker import Faker
from datetime import datetime, timedelta

class FixedIncomeTrainingGenerator:
    def __init__(self):
        self.fake = Faker()
        self.financial_currencies = ["USD", "INR", "EUR", "GBP", "JPY"]
        self.portfolio_types = ["Corporate Bonds Portfolio", "Sovereign Bonds Portfolio", "Municipal Bonds Portfolio"]
        self.schema_tables = ["Portfolios", "FixedIncomeSecurities", "PortfolioHoldings", "Valuations", "OCIChanges"]
        self.schema_columns = [
            "PortfolioId", "PortfolioName", "Description", "SecurityId", "ISIN",
            "CouponRate", "MaturityDate", "CurrencyCode", "PurchaseDate", "Quantity",
            "AmortizedCost", "UnrealizedGainLoss"
        ]
        self.examples_values = (
            "Portfolio Names: Corporate Bonds Portfolio, Sovereign Bonds Portfolio | "
            "Currencies: USD, INR, EUR, GBP, JPY"
        )

    def _random_date(self, start_year=2010, end_year=2030):
        """Generate random date between two years."""
        start_date = datetime(start_year, 1, 1)
        end_date = datetime(end_year, 12, 31)
        return self.fake.date_between(start_date=start_date, end_date=end_date)

    def _generate_one(self):
        """Generate one natural language query and SQL pair."""
        choice = random.choice([
            "simple_portfolio_filter",
            "date_filter",
            "coupon_filter",
            "currency_filter",
            "unrealized_gain_loss_filter",
            "maturity_date_filter",
            "aggregated_coupon_by_currency"
        ])

        if choice == "simple_portfolio_filter":
            portfolio = random.choice(self.portfolio_types)
            return {
                "natural_language": f"List all holdings in {portfolio}",
                "sql_query": (
                    f"SELECT * FROM PortfolioHoldings ph "
                    f"JOIN Portfolios p ON ph.PortfolioId = p.PortfolioId "
                    f"WHERE p.PortfolioName LIKE '%{portfolio.split()[0]}%'"
                )
            }

        elif choice == "date_filter":
            date = self._random_date(2020, 2025)
            return {
                "natural_language": f"Show holdings purchased after {date}",
                "sql_query": (
                    f"SELECT * FROM PortfolioHoldings "
                    f"WHERE PurchaseDate > '{date}'"
                )
            }

        elif choice == "coupon_filter":
            rate = random.uniform(2.0, 8.0)
            return {
                "natural_language": f"Find securities with coupon rate greater than {rate:.2f}%",
                "sql_query": (
                    f"SELECT * FROM FixedIncomeSecurities "
                    f"WHERE CouponRate > {rate:.2f}"
                )
            }

        elif choice == "currency_filter":
            currency = random.choice(self.financial_currencies)
            return {
                "natural_language": f"List securities with currency {currency}",
                "sql_query": (
                    f"SELECT * FROM FixedIncomeSecurities "
                    f"WHERE CurrencyCode = '{currency}'"
                )
            }

        elif choice == "unrealized_gain_loss_filter":
            threshold = random.randint(10000, 500000)
            return {
                "natural_language": f"Show holdings where unrealized gain or loss exceeds {threshold}",
                "sql_query": (
                    f"SELECT h.*, o.UnrealizedGainLoss "
                    f"FROM PortfolioHoldings h JOIN OCIChanges o ON h.HoldingId = o.HoldingId "
                    f"WHERE o.UnrealizedGainLoss > {threshold}"
                )
            }

        elif choice == "maturity_date_filter":
            date = self._random_date(2025, 2035)
            return {
                "natural_language": f"Find securities maturing after {date}",
                "sql_query": (
                    f"SELECT * FROM FixedIncomeSecurities "
                    f"WHERE MaturityDate > '{date}'"
                )
            }

        elif choice == "aggregated_coupon_by_currency":
            return {
                "natural_language": "Show average coupon rate grouped by currency",
                "sql_query": (
                    "SELECT CurrencyCode, AVG(CouponRate) AS AvgCouponRate "
                    "FROM FixedIncomeSecurities "
                    "GROUP BY CurrencyCode"
                )
            }

        else:
            raise ValueError("Unexpected query type!")

    def generate_training_data(self, num_examples=5000):
        """Generate full training set."""
        training_data = []

        seen_nl = set()  # to remove exact duplicate NL queries

        while len(training_data) < num_examples:
            example = self._generate_one()
            if example['natural_language'] not in seen_nl:
                training_data.append(example)
                seen_nl.add(example['natural_language'])

        df = pd.DataFrame(training_data)
        return df

    def save_training_data(self, df: pd.DataFrame, output_file: str):
        """Save to CSV."""
        df.to_csv(output_file, index=False)
        print(f"✅ Saved {len(df)} examples to {output_file}")

if __name__ == "__main__":
    generator = FixedIncomeTrainingGenerator()
    df = generator.generate_training_data(num_examples=5000)  # You can change number here
    generator.save_training_data(df, output_file="smartest_compact_training_dataset.csv")
