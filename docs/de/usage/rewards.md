# Belohnungen (Rewards)

## Belohnungswerte

| Ereignis | Belohnung | Terminierend |
|-------|--------|----------|
| Jeder Schritt (Standard) | -1 | Nein |
| Vom Geist gefangen | -50 | **Ja** |
| Ziel erreicht | +100 | **Ja** |

## Strategie

Um die Belohnung zu maximieren, sollte der Agent:  
1. Den kürzesten Weg zum Ziel finden.  
2. Den Geist unbedingt vermeiden.  

### Verwendung in Adversarial Search

Die oben genannten Belohnungswerte werden von den **Minimax**- und **Expectimax**-Agenten genutzt, um die Güte eines zukünftigen Zustands zu bewerten. Die Heuristik-Funktion dieser Agenten berechnet den erwarteten kumulativen Reward, um die Aktion zu wählen, die den minimalen Verlust (Minimax) oder den maximalen erwarteten Gewinn (Expectimax) verspricht.
