# ImagesMerger()

Questa classe si occupa di unire in un unica immagine diverse gadf o gasf in modo da poter essere passate successivamente alla rete neurale. 
Il costruttore è vuoto.  


# run()
Lancia lo script per unire le immagini.
I parametri passati sono fondamentali per capire da dove leggere le immagini originali, quali segnali utilizzare e come sistemare i vari pezzi del "puzzle" nella nuova immagine. 



**Parametri**:

- **input_folders**: _string_
- **resolutions**: __list__ 
- **signals**: __list__ 
- **positions**: __list__ 
- **type**: _string_ ['gadf', 'gasf']
- **img_size**: __list__ 
- **output_path**: _string_

**Esempio**: 

In questo esempio stiamo passando come dataset in input `sp500`.

Si sta scegliendo di unire in una sola immagine `4` blocchi `40x40` prendendo dalle 4 sottocartelle `1hour`, `1day`, `4hours`, `8hours` che semanticamente significa unire 4 immagini contenenti le 40 ore precedenti, 40 giorni, 40 ore raggrupate per 4 ore e 40 ore raggruppate per 8 ore precedenti. Si sta scegliendo di utilizzare soltanti i segnali `delta` e `volume`,

Il parametro `positions` è una lista di tuple, in cui ogni elemento specifica dove si deve collocare una delle 4 immagini da unire. In questo caso la prima si posizionerà in alto a sinistra, la seconda in alto a destra, la terza in basso a sinistra e l'ultima in basso a destra. I numeri rappresentano l'offset in pixel.

Il parametro `img_size` specifica la dimensione dell'immagine finale in output. In questo caso, visto che stiamo unendo 4 immagini `40x40`, l'immagine risultante sarà `80x80`.

`output_path` specifica dove salvare le immagini. Alla stringa passata come parametro ci si aggiungerà anche la cartella principale in cui vengono salvati tutti i merge e la tipologia specificata nel parametro `type`. In questo caso l'output finale sarà quindi: `../images/merge/test_merge/gasf/`.

```python

from classes import ImagesMerger

imgmerger = ImagesMerger.ImagesMerger()

imgmerger.run(input_folders='sp500',
              resolutions=['1hour', '1day', '4hours', '8hours'],
              signals=['delta', 'volume'],
              positions=[(0, 0), (40, 0), (0, 40), (40, 40)],
              type='gasf',
              img_size=[80, 80],
              output_path="test_merge_")

```