# Daily Health Check Runbook

## Overview
This runbook provides procedures for performing daily health checks on the Heart Disease Prediction system.

## Schedule
- **Frequency**: Daily
- **Recommended Time**: 09:00 AM local time
- **Duration**: ~15 minutes
- **Priority**: High

## Prerequisites
- Access to monitoring dashboard
- API access credentials
- Database access
- Logging system access

## Procedure

### 1. System Health Check

```bash
# Check API health
curl -X GET https://api.example.com/health

# Verify all endpoints
python scripts/validate_api.py --check-all-endpoints

# Check service status
systemctl status heart-disease-api
systemctl status heart-disease-worker
```

Expected Results:
- All endpoints return 200 OK
- Services show "active (running)"
- No error logs in last 24h

### 2. Model Performance Check

```bash
# Check model metrics
python scripts/validate_model_performance.py \
  --last-24h \
  --threshold-accuracy 0.80 \
  --threshold-auc 0.85

# Verify prediction latency
python scripts/check_prediction_latency.py \
  --max-latency 200
```

Expected Results:
- Accuracy ≥ 80%
- AUC ≥ 0.85
- Average latency < 200ms
- No prediction timeouts

### 3. Resource Utilization

```bash
# Check system resources
python scripts/check_resources.py \
  --cpu-threshold 80 \
  --memory-threshold 85 \
  --disk-threshold 90
```

Monitor:
- CPU usage < 80%
- Memory usage < 85%
- Disk usage < 90%
- Network I/O normal

### 4. Data Validation

```bash
# Validate recent data
python scripts/validate_data.py \
  --last-24h \
  --check-distributions \
  --check-missing
```

Check:
- Data completeness
- Distribution drift
- Missing values
- Data quality metrics

### 5. Security Check

```bash
# Check security metrics
python scripts/security_check.py \
  --check-auth-logs \
  --check-rate-limits \
  --check-failed-attempts
```

Verify:
- No unusual access patterns 
- Rate limits not exceeded
- Failed auth attempts normal
- Security logs clean

## Common Issues

### High Response Time
1. Check system resources
2. Review active connections
3. Check database performance
4. Review cache hit rate

### Model Performance Drop
1. Check input data quality
2. Verify feature distributions
3. Review recent predictions
4. Check for data drift

### Resource Issues
1. Review active processes
2. Check for memory leaks
3. Verify disk usage
4. Monitor network usage

## Escalation Procedures

### When to Escalate
- Response time > 500ms
- Accuracy < 75%
- Resource usage > 90%
- Security incidents

### Escalation Path
1. On-call engineer
2. System administrator
3. ML team lead
4. Project manager

## Reporting

### Daily Report
Generate daily health report:
```bash
python scripts/generate_health_report.py \
  --last-24h \
  --output health_report.pdf
```

Include:
- System metrics
- Model performance
- Resource utilization
- Incident summary

### Incident Documentation
For any issues:
1. Document in issue tracker
2. Update runbook if needed
3. Schedule review if recurring
4. Update monitoring if required

## Checklist

- [ ] API health verified
- [ ] Model performance checked
- [ ] Resources within limits
- [ ] Data validated
- [ ] Security verified
- [ ] Report generated
- [ ] Issues documented
- [ ] Runbook updated if needed

## References

- [Monitoring Dashboard](https://monitoring.example.com)
- [Alert Configuration](../monitoring/alerts.md)
- [Incident Response](../incidents/response.md)
- [Performance Baselines](../monitoring/baselines.md)

