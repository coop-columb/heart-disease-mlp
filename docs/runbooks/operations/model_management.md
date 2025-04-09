# Model Management Runbook

## Overview
This runbook details procedures for managing ML models in the Heart Disease Prediction system, including monitoring, updating, and maintaining model quality.

## Model Lifecycle

### 1. Model Monitoring

#### Daily Checks
```bash
# Check model performance
python scripts/check_model_metrics.py \
  --last-24h \
  --metrics "accuracy,auc,f1" \
  --thresholds "0.80,0.85,0.75"

# Monitor prediction distribution
python scripts/analyze_predictions.py \
  --distribution-check \
  --drift-threshold 0.1

# Check data quality
python scripts/validate_input_data.py \
  --check-distributions \
  --check-missing
```

#### Weekly Analysis
```bash
# Comprehensive analysis
python scripts/analyze_model_performance.py \
  --last-7d \
  --full-metrics \
  --generate-report

# Check for data drift
python scripts/check_data_drift.py \
  --compare-to baseline \
  --all-features

# Generate performance report
python scripts/generate_model_report.py \
  --last-7d \
  --include-plots
```

### 2. Model Updates

#### Preparation
```bash
# Prepare training data
python scripts/prepare_training_data.py \
  --validate-data \
  --check-balance

# Validate features
python scripts/validate_features.py \
  --check-distributions \
  --check-correlations

# Setup training environment
python scripts/setup_training_env.py \
  --gpu-check \
  --memory-check
```

#### Training
```bash
# Train new model
python scripts/train_model.py \
  --config configs/training.yml \
  --validate \
  --save-artifacts

# Evaluate model
python scripts/evaluate_model.py \
  --comprehensive \
  --compare-baseline \
  --generate-report

# Validate model
python scripts/validate_model.py \
  --check-performance \
  --check-bias \
  --check-robustness
```

#### Deployment
```bash
# Deploy new model
python scripts/deploy_model.py \
  --version <version> \
  --environment staging \
  --backup-current

# Verify deployment
python scripts/verify_deployment.py \
  --run-tests \
  --check-performance

# Monitor deployment
python scripts/monitor_deployment.py \
  --duration 1h \
  --alert-threshold 0.1
```

### 3. Model Maintenance

#### Regular Tasks
```bash
# Backup model artifacts
python scripts/backup_models.py \
  --include-weights \
  --include-config

# Clean old versions
python scripts/clean_old_models.py \
  --keep-last 5 \
  --keep-best 2

# Update documentation
python scripts/update_model_docs.py \
  --current-version \
  --include-metrics
```

#### Performance Optimization
```bash
# Optimize model
python scripts/optimize_model.py \
  --target latency \
  --max-loss 0.01

# Benchmark performance
python scripts/benchmark_model.py \
  --scenarios all \
  --load-test

# Update configurations
python scripts/update_model_config.py \
  --optimize-params \
  --validate
```

## Model Quality Assurance

### 1. Performance Metrics
Monitor:
- Accuracy (≥ 80%)
- AUC-ROC (≥ 0.85)
- F1 Score (≥ 0.75)
- Precision (≥ 0.80)
- Recall (≥ 0.75)

### 2. Data Quality
Check:
- Missing values (< 5%)
- Feature distributions
- Data drift (< 10%)
- Feature correlations
- Input validation

### 3. Bias Monitoring
Verify:
- Protected attributes
- Prediction fairness
- Group metrics
- Bias metrics

## Troubleshooting

### 1. Performance Degradation
```bash
# Analyze degradation
python scripts/analyze_degradation.py \
  --last-24h \
  --full-metrics

# Check data quality
python scripts/check_data_quality.py \
  --recent-inputs \
  --validate

# Review predictions
python scripts/review_predictions.py \
  --failed-only \
  --analyze
```

### 2. Prediction Issues
```bash
# Debug predictions
python scripts/debug_predictions.py \
  --last-100 \
  --analyze-errors

# Validate inputs
python scripts/validate_inputs.py \
  --check-ranges \
  --check-types

# Test specific cases
python scripts/test_predictions.py \
  --use-test-cases \
  --verbose
```

### 3. Resource Issues
```bash
# Check resources
python scripts/check_model_resources.py \
  --memory-usage \
  --cpu-usage \
  --gpu-usage

# Optimize performance
python scripts/optimize_performance.py \
  --target resources \
  --maintain-accuracy

# Monitor usage
python scripts/monitor_resources.py \
  --continuous \
  --alert-threshold 0.9
```

## Best Practices

### 1. Model Updates
- Regular evaluation schedule
- Comprehensive testing
- Gradual rollout
- Backup procedures
- Documentation updates

### 2. Quality Control
- Automated testing
- Manual review
- Performance thresholds
- Bias checking
- Security validation

### 3. Documentation
- Model cards
- Version history
- Performance metrics
- Known issues
- Update logs

## Checklist

### Daily Tasks
- [ ] Check performance metrics
- [ ] Monitor predictions
- [ ] Validate data quality
- [ ] Review alerts
- [ ] Update logs

### Weekly Tasks
- [ ] Comprehensive analysis
- [ ] Check for drift
- [ ] Generate reports
- [ ] Review documentation
- [ ] Backup artifacts

### Monthly Tasks
- [ ] Full evaluation
- [ ] Update baselines
- [ ] Clean old versions
- [ ] Review procedures
- [ ] Update documentation

## References

- [Model Architecture](../../docs/model.md)
- [Training Guide](../../docs/training.md)
- [Evaluation Metrics](../../docs/metrics.md)
- [Deployment Guide](../../docs/deployment.md)

