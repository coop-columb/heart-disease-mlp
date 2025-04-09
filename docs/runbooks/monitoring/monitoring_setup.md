# Monitoring and Alerting Runbook

## Overview
This runbook details the monitoring and alerting configuration for the Heart Disease Prediction system, ensuring system health, model performance, and issue detection.

## Monitoring Components

### 1. System Metrics

#### Infrastructure Monitoring
```yaml
# Prometheus configuration
metrics:
  - name: cpu_usage
    type: gauge
    threshold: 80
    duration: 5m
    
  - name: memory_usage
    type: gauge
    threshold: 85
    duration: 5m
    
  - name: disk_usage
    type: gauge
    threshold: 90
    duration: 5m
    
  - name: network_io
    type: counter
    threshold: null  # Alert on sudden changes
```

#### API Monitoring
```yaml
# API metrics
metrics:
  - name: request_latency
    type: histogram
    threshold: 200  # ms
    percentile: 95
    
  - name: error_rate
    type: gauge
    threshold: 0.01  # 1%
    duration: 5m
    
  - name: request_rate
    type: counter
    threshold: null  # Alert on anomalies
```

### 2. Model Metrics

#### Performance Monitoring
```yaml
# Model metrics
metrics:
  - name: prediction_accuracy
    type: gauge
    threshold: 0.80
    window: 24h
    
  - name: auc_score
    type: gauge
    threshold: 0.85
    window: 24h
    
  - name: prediction_latency
    type: histogram
    threshold: 100  # ms
    percentile: 95
```

#### Data Quality
```yaml
# Data metrics
metrics:
  - name: missing_values
    type: gauge
    threshold: 0.05  # 5%
    
  - name: data_drift
    type: gauge
    threshold: 0.1  # 10%
    window: 24h
    
  - name: feature_correlation
    type: gauge
    threshold: 0.9
```

### 3. Security Monitoring

#### Access Monitoring
```yaml
# Security metrics
metrics:
  - name: failed_auth
    type: counter
    threshold: 10
    duration: 5m
    
  - name: rate_limit_hits
    type: counter
    threshold: 100
    duration: 5m
    
  - name: suspicious_patterns
    type: gauge
    threshold: 0.8
```

## Alert Configuration

### 1. Severity Levels

```yaml
# Alert severity configuration
severities:
  critical:
    page_oncall: true
    notification_channels: ["pager", "slack", "email"]
    auto_incident: true
    
  high:
    page_oncall: true
    notification_channels: ["slack", "email"]
    auto_incident: false
    
  medium:
    page_oncall: false
    notification_channels: ["slack"]
    auto_incident: false
    
  low:
    page_oncall: false
    notification_channels: ["email"]
    auto_incident: false
```

### 2. Alert Rules

```yaml
# System alerts
alerts:
  - name: high_cpu_usage
    metric: cpu_usage
    threshold: 80
    duration: 5m
    severity: high
    
  - name: high_error_rate
    metric: error_rate
    threshold: 0.01
    duration: 5m
    severity: critical
    
  - name: model_performance_drop
    metric: prediction_accuracy
    threshold: 0.80
    window: 24h
    severity: high
```

### 3. Notification Channels

```yaml
# Notification configuration
channels:
  slack:
    webhook: "https://hooks.slack.com/services/..."
    channels:
      - "#monitoring-alerts"
      - "#oncall"
      
  email:
    from: "alerts@example.com"
    to: ["oncall@example.com"]
    
  pager:
    service: "pagerduty"
    integration_key: "..."
```

## Dashboard Configuration

### 1. System Dashboard
```yaml
# Grafana dashboard
panels:
  - name: System Overview
    metrics:
      - cpu_usage
      - memory_usage
      - disk_usage
    refresh: 1m
    
  - name: API Performance
    metrics:
      - request_latency
      - error_rate
      - request_rate
    refresh: 30s
```

### 2. Model Dashboard
```yaml
# Model performance dashboard
panels:
  - name: Model Metrics
    metrics:
      - prediction_accuracy
      - auc_score
      - f1_score
    refresh: 5m
    
  - name: Prediction Analysis
    metrics:
      - prediction_latency
      - prediction_distribution
      - confidence_scores
    refresh: 5m
```

## Monitoring Procedures

### 1. Setting Up Monitoring

```bash
# Deploy monitoring stack
python scripts/deploy_monitoring.py \
  --prometheus \
  --grafana \
  --alertmanager

# Configure metrics
python scripts/configure_metrics.py \
  --config monitoring/metrics.yml

# Setup dashboards
python scripts/setup_dashboards.py \
  --import-templates
```

### 2. Alert Configuration

```bash
# Configure alerts
python scripts/configure_alerts.py \
  --rules alerts/rules.yml \
  --channels alerts/channels.yml

# Test notifications
python scripts/test_alerts.py \
  --all-channels \
  --verify
```

### 3. Verification

```bash
# Verify monitoring
python scripts/verify_monitoring.py \
  --check-metrics \
  --check-alerts

# Test alert pipeline
python scripts/test_alert_pipeline.py \
  --generate-incidents \
  --verify-notifications
```

## Maintenance

### 1. Regular Tasks

```bash
# Update dashboards
python scripts/update_dashboards.py \
  --refresh-templates \
  --backup-current

# Clean old data
python scripts/clean_metrics.py \
  --older-than 90d \
  --backup

# Verify configuration
python scripts/verify_config.py \
  --check-all
```

### 2. Alert Tuning

```bash
# Analyze alert history
python scripts/analyze_alerts.py \
  --last-30d \
  --generate-report

# Update thresholds
python scripts/update_thresholds.py \
  --based-on-history \
  --min-confidence 0.95
```

## Troubleshooting

### 1. Missing Metrics
```bash
# Check metric collection
python scripts/check_metrics.py \
  --list-missing \
  --verify-pipeline

# Verify exporters
python scripts/verify_exporters.py \
  --all \
  --verbose
```

### 2. Alert Issues
```bash
# Test alert pipeline
python scripts/test_alerts.py \
  --generate-test \
  --verify-delivery

# Check notification channels
python scripts/check_notifications.py \
  --all-channels \
  --send-test
```

## Best Practices

### 1. Metric Collection
- Use appropriate metric types
- Set meaningful thresholds
- Include context
- Monitor trends
- Regular validation

### 2. Alert Configuration
- Clear severity levels
- Actionable alerts
- Proper routing
- Documented procedures
- Regular review

### 3. Dashboard Design
- Clear visibility
- Logical grouping
- Key metrics prominent
- Proper time ranges
- Useful annotations

## References

- [Prometheus Docs](https://prometheus.io/docs/)
- [Grafana Docs](https://grafana.com/docs/)
- [Alert Manager Docs](https://prometheus.io/docs/alerting/latest/alertmanager/)
- [Monitoring Best Practices](../../docs/monitoring.md)

