# Assignment 1 - ANN, CNN och transfer learning

Det här repot innehåller min lösning på Assignment 1. Arbetet börjar från grunden, med en enkel neuron och ANN-lager, och går sedan vidare till CNN-modeller, MLOps-delar, data augmentation och till sist transfer learning med ResNet50.

Jag har fölt instruktioner från [notin](https://spiking-thai2.notion.site/Shared-Uppgift-1-Perceptron-fo-r-OCR-2b9a7700ac26801d9a3ad9eac0e87b82) så gott jag har kunnat. Det gjorde framför allt NumPy/PyTorch-delen extra lärorik, eftersom det blev tydligare vad PyTorch faktiskt hjälper till med när man sedan använder färdiga lager, optimerare och backpropagation.

## Notebookar

### `assignment1-part-1.ipynb`

Den första notebooken handlar om grunderna bakom neurala nätverk.

Här implementeras först en enkel neuron utan NumPy, alltså mer eller mindre bara med vanliga räkneoperationer. Efter det byggs samma idé vidare med NumPy, där ett helt lager kan beräknas med matrismultiplikation istället för att hantera varje neuron separat.

I den sista delen används PyTorch för att bygga och träna en ANN på MNIST. Där blir skillnaden ganska tydlig mellan att själv skriva beräkningarna och att låta PyTorch hantera modell, gradients, optimizer och träning. Notebooken körs också med CUDA/GPU om det finns tillgängligt.

Det här var den del jag tyckte var mest lärorik, eftersom den gjorde kopplingen mellan teori, NumPy och PyTorch mer konkret.

### `assignment1-part-2.ipynb`

Den andra notebooken bygger vidare från ANN till CNN på MNIST.

Här finns flera CNN-arkitekturer:

- `SmallCNN`
- `MediumCNN`
- `LargeCNN`

Modellerna jämförs med hjälp av tränings- och valideringsresultat. Notebooken innehåller även data augmentation, regularisering med BatchNorm, Dropout och weight decay, samt sparade checkpoints för bästa modell.

Jag lade också till mer MLOps-liknande delar, till exempel:

- run-mappar för varje träning
- sparade config-filer
- checkpoints
- TensorBoard-loggning
- visualisering av loss, accuracy, confusion matrix och felklassificerade exempel

I slutet finns även hyperparameter tuning med en mer flexibel CNN (`FlexCNN`), där olika kombinationer av lager, kernel sizes, activation functions, dropout, learning rate, batch size och augmentation testas.

Det svåraste i den här delen var att hålla koll på tensor shapes. Särskilt i CNN-modeller märks det snabbt om man har räknat fel på bildstorleken efter convolution/pooling, eftersom det påverkar storleken på första fully connected-lagret. Overfitting var också en utmaning, eftersom vissa modeller snabbt blir bra på träningsdata men inte generaliserar lika bra.

### `assignment1-part-3.1.ipynb`

Den här notebooken byter dataset från MNIST till Signs Detection Dataset från Kaggle.

Datasetet laddas ner med `kagglehub` och läses från HDF5-filer med `h5py`. Notebooken undersöker först strukturen i HDF5-filen och visualiserar exempelbilder, innan en egen `SignsDataset`-klass byggs för PyTorch.

Sedan tränas en egen CNN (`MediumCNN`) från grunden på handteckensbilderna. Det ingår även data augmentation, till exempel rotation, crop, färgvariation och normalisering. Resultatet utvärderas med träningskurvor, testaccuracy, per-class accuracy och confusion matrix.

Den här delen var nyttig eftersom den var närmare ett "riktigt" dataset än MNIST. Det blev mer arbete med datasetformat, preprocessing och augmentation innan själva modellen ens kunde tränas.

### `assignment1-part-3.2.ipynb`

Den sista notebooken använder transfer learning med ResNet50 på Signs-datasetet.

Först laddas en ResNet50-modell med förtränade ImageNet-vikter. Eftersom ResNet50 förväntar sig ImageNet-format skalas och normaliseras bilderna enligt ResNet50-pipelinen.

Träningen görs i två steg:

1. Först fryses ResNet50-backbone och bara det sista klassificeringslagret tränas.
2. Sedan låses `layer4` upp och fine-tunas med lägre learning rate.

Det här var också en av de mest intressanta delarna. Det blev ganska tydligt varför transfer learning är så användbart när datasetet är litet. Modellen behöver inte lära sig kanter, former och enklare visuella mönster från början, utan kan återanvända features från ImageNet och bara anpassa slutet av nätverket till handtecknen.

Jag är särskilt nöjd med att ResNet50 blev så mycket bättre efter fine-tuning. Den egna CNN-modellen fungerade ändå okej, men ResNet50 fick ett tydligt lyft när sista blocket fick anpassas till Signs-datasetet.

## Resultat i korthet

I part 3 jämförs den egna CNN-modellen med ResNet50:

| Modell | Testaccuracy | Kommentar |
| --- | ---: | --- |
| Egen MediumCNN | 86,7 % | Tränad från grunden |
| ResNet50 fryst | 87,5 % | Snabb träning, bara sista lagret tränas |
| ResNet50 fine-tuned | 97,5 % | Bäst resultat efter att sista ResNet-blocket låsts upp |

Det mest intressanta här är att den frysta ResNet50-modellen redan ligger ungefär på samma nivå som den egenbyggda CNN-modellen, trots att nästan hela modellen är fryst. När `layer4` sedan fine-tunas blir skillnaden mycket större.

## Installation

Skapa gärna en virtuell miljö först:

```bash
python -m venv .venv
source .venv/bin/activate
```

Installera sedan beroenden:

```bash
pip install -r requirements.txt
```

PyTorch och torchvision kan behöva installeras separat beroende på om man kör CPU eller CUDA. Använd rätt kommando från PyTorchs installationssida för din miljö.

## Reflektion

Det jag tar med mig mest från uppgiften är hur mycket som faktiskt händer mellan en enkel neuron och en färdig CNN eller ResNet-modell. När man bygger små delar själv med NumPy blir det lättare att förstå varför PyTorchs abstraktioner är användbara.

Samtidigt var det ganska lätt att gå vilse i tensor shapes och modellstorlekar. En liten ändring i convolution eller pooling kan göra att nästa lager inte längre passar. Overfitting var också något som behövde hanteras med augmentation, dropout, weight decay och valideringskurvor.

Transfer learning var nog den del som överraskade mest positivt. Det kändes först lite som att "fuska" att använda en färdig ResNet50, men resultatet visar ganska bra varför det är ett vanligt arbetssätt i praktiken. Man återanvänder generell bildförståelse och lägger kraften på att anpassa modellen till det specifika problemet.
