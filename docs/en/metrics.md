# Documentation Metrics

This page shows quality metrics for the documentation and code.

## API Coverage (Docstrings)

![interrogate](../assets/interrogate.svg)

## Status Overview

<div id="metrics-dashboard">
  Loading metrics...
</div>

<script>
fetch('../assets/metrics.json')
  .then(response => response.json())
  .then(data => {
    const dashboard = document.getElementById('metrics-dashboard');
    dashboard.innerHTML = `
      <table>
        <tr><td>Last CI Run</td><td>${new Date(data.last_run).toLocaleString()}</td></tr>
      </table>
    `;
  })
  .catch(err => {
    document.getElementById('metrics-dashboard').innerText = 'Metrics currently unavailable.';
  });
</script>
