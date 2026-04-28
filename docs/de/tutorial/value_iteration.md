# Value Iteration Tutorial

Value Iteration ist ein klassischer Dynamic Programming Algorithmus, der verwendet wird, um die optimale Value Function $v^*$ zu finden. In diesem Projekt nutzen wir ihn zur präzisen Pfadplanung.

## Konzept

Bei der Value Iteration berechnen wir iterativ den Wert jedes Zustands (Zelle), basierend auf den Belohnungen der benachbarten Zustände und deren eigenen Werten.

Die Bellman-Gleichung ist das Herzstück:
$$V(s) \leftarrow \max_a \sum_{s', r} p(s', r | s, a) [r + \gamma V(s')]$$

In unserem Grid bedeutet das:  
- **Ziel**: Hoher positiver Wert (100).  
- **Normaler Schritt**: Kleiner negativer Wert (-1).  
- **Hindernisse/Wände**: Verhindern Bewegung (Zustandswert bleibt niedrig).  

## Interaktives Notebook

Du kannst den Algorithmus Schritt für Schritt in unserem interaktiven Jupyter Notebook nachvollziehen und trainieren:

[![Open In Colab](../../assets/colab-badge.svg)](https://colab.research.google.com/github/dgaida/adversarial2dEnvAI/blob/master/notebooks/Value_Iteration.ipynb)

## Implementierung im Projekt

Im Code findest du die Implementierung in:  
- `src/custom_grid_env/planner.py`: Hier liegt die Logik für `value_iteration()`.  
- `src/custom_grid_env/agents/value_iteration_agent.py`: Der Agent, der diese Werte nutzt, um die beste Aktion zu wählen.  
