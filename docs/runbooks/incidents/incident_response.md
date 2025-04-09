# Incident Response Runbook

## Overview
This runbook provides procedures for responding to incidents in the Heart Disease Prediction system. It ensures quick, effective response to service disruptions, performance issues, or security events.

## Incident Severity Levels

### Severity 1 (Critical)
- System unavailable
- Incorrect predictions
- Data breach
- Response time > 1s
- Multiple alerts firing

### Severity 2 (High)
- Degraded performance
- Model accuracy drop
- High error rates
- Resource warnings
- Security warnings

### Severity 3 (Medium)
- Minor performance issues
- Single component issues
- Non-critical errors
- Warning alerts

### Severity 4 (Low)
- Cosmetic issues
- Minor bugs
- Single error alerts
- Performance warnings

## Initial Response

### 1. Assessment
```bash
# Quick system check
python scripts/system_check.py --quick

# Check critical metrics
python scripts/check_critical_metrics.py

# Review recent alerts
python scripts/review_alerts.py --last-1h
```

Gather:
- Incident start time
- Affected components
- Impact scope
- Current status

### 2. Communication
1. **Internal**
   - Alert on-call team
   - Update status page
   - Notify stakeholders

2. **External** (if needed)
   - Update status page
   - Prepare user communication
   - Contact affected clients

### 3. Immediate Actions

#### API Issues
```bash
# Check API status
curl -X GET https://api.example.com/health

# Review error rates
python scripts/check_error_rates.py --last-15m

# Check recent logs
python scripts/analyze_logs.py --last-15m
```

#### Model Issues
```bash
# Validate model performance
python scripts/validate_model_performance.py --quick

# Check prediction logs
python scripts/analyze_predictions.py --last-15m

# Verify data quality
python scripts/validate_recent_data.py
```

#### Resource Issues
```bash
# Check resource usage
python scripts/check_resources.py --all

# Analyze system load
python scripts/analyze_load.py --detailed

# Monitor network
python scripts/check_network.py --full
```

## Incident Response Procedures

### 1. Service Disruption

```bash
# Verify service status
systemctl status heart-disease-api
systemctl status heart-disease-worker

# Check dependencies
python scripts/check_dependencies.py --all

# Review error logs
python scripts/analyze_logs.py --error-only
```

Actions:
1. Identify failing component
2. Check dependencies 
3. Review recent changes
4. Consider rollback

### 2. Performance Degradation

```bash
# Analysis
python scripts/analyze_performance.py \
  --last-1h \
  --check-all-components

# Resource check
python scripts/check_resources.py --detailed

# Load testing
python scripts/quick_load_test.py
```

Actions:
1. Identify bottleneck
2. Check resource usage
3. Review recent traffic
4. Consider scaling

### 3. Model Issues

```bash
# Validate predictions
python scripts/validate_recent_predictions.py

# Check data quality
python scripts/validate_input_data.py

# Analyze model metrics
python scripts/analyze_model_metrics.py --last-24h
```

Actions:
1. Verify input data
2. Check model metrics
3. Review recent changes
4. Consider model rollback

### 4. Security Incidents

```bash
# Security check
python scripts/security_audit.py --quick

# Review access logs
python scripts/analyze_access_logs.py --last-24h

# Check authentication
python scripts/verify_auth_logs.py
```

Actions:
1. Isolate affected systems
2. Review security logs
3. Block suspicious activity
4. Contact security team

## Recovery Procedures

### 1. Service Recovery
```bash
# Health check
python scripts/health_check.py --full

# Verify functionality
python scripts/verify_functionality.py --all

# Test critical paths
python scripts/test_critical_paths.py
```

### 2. Model Recovery
```bash
# Restore previous model
python scripts/restore_model.py --version <version>

# Validate performance
python scripts/validate_model.py --full

# Monitor predictions
python scripts/monitor_predictions.py --alert
```

### 3. Security Recovery
```bash
# Security scan
python scripts/security_scan.py --full

# Reset credentials
python scripts/rotate_credentials.py

# Update security rules
python scripts/update_security.py
```

## Post-Incident

### 1. Documentation
- Incident timeline
- Actions taken
- Root cause
- Recovery steps
- Lessons learned

### 2. Analysis
- Review metrics
- Analyze logs
- Check patterns
- Identify improvements

### 3. Prevention
- Update monitoring
- Adjust thresholds
- Improve automation
- Update runbooks

## Escalation Path

### Technical Issues
1. On-call Engineer
2. System Administrator
3. Technical Lead
4. CTO

### Security Issues
1. Security Team
2. Security Lead
3. CISO
4. Legal Team

## Communication Templates

### Status Updates
```
[Severity] Incident Update
Status: [Investigating/Identified/Resolving/Resolved]
Impact: [Service/Performance/Security]
Details: [Brief description]
Next Update: [Time]
```

### Resolution Notice
```
Incident Resolution
Time: [Resolution time]
Duration: [Total time]
Impact: [Summary]
Root Cause: [Brief explanation]
Prevention: [Future measures]
```

## Checklist

### Initial Response
- [ ] Assess severity
- [ ] Alert appropriate teams
- [ ] Begin investigation
- [ ] Update status page

### During Incident
- [ ] Regular updates
- [ ] Log all actions
- [ ] Monitor progress
- [ ] Update stakeholders

### Resolution
- [ ] Verify recovery
- [ ] Document incident
- [ ] Update runbooks
- [ ] Schedule review

## References

- [Monitoring Guide](../monitoring/README.md)
- [Alert Configuration](../monitoring/alerts.md)
- [Recovery Procedures](../recovery/README.md)
- [Security Policies](../security/README.md)

