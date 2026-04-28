# Dokumentations-Metriken

Diese Seite zeigt Qualitätsmetriken für die Dokumentation und den Code.

## API-Abdeckung (Docstrings)

![interrogate](assets/interrogate.svg)

## Status-Übersicht

<div id="metrics-dashboard">
  Lade Metriken...
</div>

<script>
fetch('assets/metrics.json')
  .then(response => response.json())
  .then(data => {
    const dashboard = document.getElementById('metrics-dashboard');
    dashboard.innerHTML = `
      <table>
        <tr><td>Letzter CI-Lauf</td><td>${new Date(data.last_run).toLocaleString()}</td></tr>
      </table>
    `;
  })
  .catch(err => {
    document.getElementById('metrics-dashboard').innerText = 'Metriken aktuell nicht verfügbar.';
  });
</script>
