## Testing ML Security

1. Data Leakage:
```bash
python scripts/validation/ml/check_data_leakage.py   --data-file data/processed/data.csv   --temporal-column timestamp   --target-column target   --fold-count 5   --output-path validation_results.json
```

2. Model Security:
```bash
python scripts/validation/ml/check_model_security.py   --model-path models/model.pkl   --data-path data/test_data.npz   --epsilon 0.1   --output-path security_results.json
```

## Testing API Security

1. Authentication:
```bash
python scripts/validation/security/api/check_auth_security.py   --base-url https://api.example.com   --username test_user   --password test_pass   --output-path auth_results.json
```

2. Rate Limiting:
```bash
python scripts/validation/security/api/test_rate_limiting.py   --base-url https://api.example.com   --endpoint predict   --concurrent-users 5   --output-path rate_limit_results.json
```
