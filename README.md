# Silent Burnout Detector  
### An Ethical Early-Warning System for Developer Burnout Risk

Silent Burnout Detector is a machine learning project that identifies **early signals of developer burnout risk** using **aggregate behavioral patterns**, not invasive monitoring.

The goal is **early awareness and support**, not judgment or diagnosis.

> This system estimates *risk*, not mental health conditions.

---

## Why This Project Exists

Burnout in software teams is often:
- Detected too late
- Addressed only after performance drops
- Discussed informally, without data support

Most existing tools focus on **output metrics**, not **well-being signals**.

This project explores how **non-intrusive behavioral data** can be used responsibly to surface *early warning signs*, enabling timely and human-centered intervention.

---

## Core Principles

- **Privacy-first**:  
  No message content, code content, or personal data is analyzed.

- **Risk, not diagnosis**:  
  Outputs are probabilities, not medical or psychological conclusions.

- **Explainability over opacity**:  
  Every prediction can be explained in human terms.

- **Decision support, not surveillance**:  
  The system is meant to inform conversations, not evaluate individuals.

---

## What the System Does

1. Collects (simulated) developer activity data  
2. Engineers weekly behavioral features  
3. Generates burnout risk labels using cumulative signals  
4. Trains ML models to estimate risk probability  
5. Explains *why* a developer is flagged  
6. Validates logic through exploratory analysis

---

## Behavioral Signals Used

### Work Patterns (GitHub-like data)
- Average weekly commits
- Irregularity in commit activity
- Late-night commit percentage

### Communication Patterns
- Messages sent per day
- Average response time
- Silent days (no communication)

### Workflow Friction
- Issue reopen rate
- Pull request merge delays

These are **behavioral proxies**, not performance judgments.

---

## Project Structure

 