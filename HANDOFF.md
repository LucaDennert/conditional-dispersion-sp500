# Handoff del progetto: Conditional Dispersion sull'S&P 500

*Istruzioni complete per chi prende in mano il progetto (persona o assistente AI). Stato al 13/11/2026. Repository pubblico: https://github.com/LucaDennert/conditional-dispersion-sp500*

---

## 0. Il tuo ruolo e le regole non negoziabili

Lavori con **Luca**, autore del repository. Il progetto è un paper di ricerca, con codice pubblicato su GitHub. Luca scrive in italiano e vuole risposte in italiano, chiare per un trader di desk ma rigorose per un quant. Conosce bene le opzioni dal lato del desk.

Regole:

1. **Mai pubblicare dati su GitHub.** Tutti i dati vengono da WRDS (OptionMetrics, CRSP, Fama-French) e sono sotto licenza: non vanno pubblicati né i dati grezzi, né i file intermedi, né le serie derivate in tabella (per esempio la correlazione implicita mese per mese, le IV per titolo, le serie giornaliere). Si pubblicano solo codice, documenti, statistiche aggregate e grafici. Prima di ogni commit controlla `git ls-files` e `git status`. Il `.gitignore` esclude già csv, parquet e pickle: non forzarlo.
2. **La pre-registrazione è vincolante.** `PREREGISTRATION.md` è stato committato prima di calcolare qualsiasi segnale (commit 70e103a, 12/11/2026 17:13; hash cambiato il 13/11 dalla pulizia della cronologia, data invariata). Le regole non si cambiano dopo aver visto i risultati. Ogni modifica va nella sezione *Amendments*, con data e motivo, e va approvata da Luca. I risultati con le regole originali si riportano comunque.
3. **Il periodo di test 2011-01-01 / 2025-08-29 si usa una volta sola**, per il kill test della parte 3. Tutti i parametri si stimano sul 1996–2010.
4. **Chiedi conferma a Luca** prima di qualsiasi deviazione dalla pre-registrazione e prima di pubblicare qualcosa di dubbio.
5. **Prima di dichiarare un risultato, verificalo** sui dati. Niente numeri a memoria.

---

## 1. La domanda di ricerca e la struttura del paper (concordata, tutto su SPX)

**Ipotesi.** Le opzioni sull'indice prezzano più correlazione di quella che i titoli realizzano poi (premio di correlazione; Driessen, Maenhout e Vilkov, 2009). Il dispersion trade incassa questo premio: short vol sull'indice, long vol sui titoli. Perde però quando la correlazione sale di colpo. Aprire tranche solo quando la correlazione implicita è "ricca" rispetto a un forecast della correlazione realizzata, e il regime di vol è calmo, dovrebbe migliorare il rendimento corretto per il rischio rispetto ad aprire ogni mese.

Il paper ha quattro parti:

1. **Il premio di correlazione, 1996–2025.** Correlazione implicita contro correlazione realizzata dopo, a 1 e 3 mesi, per periodo e per regime.
2. **Il dispersion trade standard.** Trade incondizionato vega-neutral con straddle: performance, costi e test sui fattori di rischio.
3. **La trade idea iniziale.** Regola condizionale pre-registrata, testata una volta sul 2011–2025 con il kill test.
4. **Il trade finale con modelli più complessi.** Va progettato e pre-registrato **prima** di eseguire il test della parte 3, così entrambi vengono giudicati una sola volta sul 2011–2025. Altrimenti è solo esplorativo. È da proporre a Luca.

Decisioni ancora aperte, spettano a Luca: formato del paper (LaTeX/Overleaf oppure documento) ed elenco degli autori.

---

## 2. Dati: cosa esiste, dove e come è stato ottenuto

**Posizione:** in una cartella locale `data_raw/`. Non sono in git.

| File | Contenuto | Fonte WRDS |
|---|---|---|
| `crsp_sp500_constituents_daily.csv.gz` (280 MB) | Membri S&P 500 giornalieri 1996–2025: PERMNO, DlyCalDt, DlyCap, DlyRet, DlyRetx, DlyPrc e altre colonne. 3,79 milioni di righe | CRSP Index v2 (CIZ), "S&P 500 Index Constituents" (`dsp500list_v2`) |
| `optionm_crsp_link.csv` | Link secid ↔ PERMNO (secid, sdate, edate, PERMNO, score). Usare score = 1 | WRDS Linking Suite, `wrdsapps.opcrsphist` |
| `optionm_stdopd_spx.csv.gz` | Opzioni standardizzate SPX (secid 108105), tutte le scadenze, call e put, con forward_price | IvyDB US `stdopd` |
| `optionm_stdopd_members.csv.gz` (197 MB) | Opzioni standardizzate dei 1.235 secid membri, days ∈ {30, 91}, call e put, impl_volatility e forward_price. 21,8 milioni di righe | IvyDB US `stdopd` |
| `optionm_secprd_spx_vix.csv.gz` | Prezzi giornalieri di SPX e VIX | IvyDB US `secprd` |
| `optionm_zerocd.csv.gz` | Curva zero OptionMetrics (date, days, rate in %) | IvyDB US `zerocd` |
| `optionm_vsurfd_spx_91d.csv.gz` | Superficie di volatilità SPX a 91 giorni, griglia in delta: impl_volatility, impl_strike, impl_premium | IvyDB US `vsurfd` |
| `ff5_umd_daily.csv` | Fattori Fama-French 5 + momentum giornalieri (mktrf, smb, hml, rmw, cma, rf, umd), in decimali | Fama-French su WRDS |
| `member_secids.csv/.txt` | Lista dei secid dei membri | Derivata |
| `optionm_zerocd.parquet` | Duplicato inutilizzato | — |

**Copertura:** IvyDB US va dal 4/1/1996 al **29/8/2025**; non esistono date successive. IvyDB Europe **non** è in licenza Bocconi.

**Validazioni già superate** (`src/validate_data.py`):
- tra 498 e 508 membri al giorno, nessun duplicato;
- portafoglio CRSP pesato per capitalizzazione contro SPX: correlazione giornaliera 0,9998, tracking error 0,4% annuo, scarto medio pari al dividend yield;
- chiusure di SPX e VIX esatte, massimo del VIX 82,69 il 16/3/2020;
- IV 91d call e put differiscono in media di 0,6 punti di vol;
- corr(IV30 ATM, VIX) = 0,991;
- copertura media delle IV dei membri: 99,7% del peso dell'indice (minimo 95,5%);
- i rendimenti oltre ±50% sono eventi reali (Lehman, WaMu, First Republic e simili).

### Il dato che manca, ed è il blocco principale

**Prezzi delle opzioni listate con bid e ask reali** (IvyDB US, `optionm.opprcdYYYY`). Servono per:
- il costo di round-trip *c* della regola;
- il motore di P&L della parte 3;
- i costi reali della parte 2.

Servono anche le **superfici di volatilità dei membri** (`vsurfd`), per gli strike model-free dei variance swap sui titoli e per la robustezza alla Linders-Schoutens.

**Come ottenerli.** Il modulo web non basta: il limite è di 4.096 caratteri di codici e non ci sono filtri su strike o scadenza relativi allo spot. Serve **Python sul WRDS JupyterHub** (wrds-jupyter.wharton.upenn.edu) con l'account WRDS personale di Luca, ora attivo. Il notebook è `notebooks/01_estrazione_ivydb_us.ipynb`. **Non è ancora mai stato eseguito**: lancia prima le celle 1–3 (elenco tabelle, colonne, secid di SPX/VIX e data finale) e correggi i nomi se non tornano. Procedura in due passi:
- **Passo A.** In ogni data d'ingresso (primo giorno di borsa del mese), per SPX e per i top-50 per capitalizzazione del mese precedente, si sceglie lo straddle con scadenza mensile standard più vicina a 91 giorni (tra 60 e 130) e strike più vicino allo spot, con call e put entrambe quotate (bid > 0, ask > bid). A parità, si prende lo spread minore.
- **Passo B.** Storia giornaliera di quei `optionid`, dall'ingresso alla scadenza.

Attenzioni:
- `strike_price` in IvyDB è moltiplicato per 1000;
- prima del 2015 le scadenze possono risultare di sabato;
- SPX ha anche le weekly (SPXW): tenere solo le mensili standard.

Se l'accesso Python non fosse disponibile, il piano B è prezzare straddle sintetici dalla `vsurfd`. Richiede un emendamento, perché la pre-registrazione specifica contratti listati comprati all'ask e venduti al bid.

**Note pratiche sul web WRDS** (accesso biblioteca tramite proxy OpenAthens Bocconi; l'account è condiviso, quindi nell'elenco query compaiono anche quelle di altri):
- liste di codici lunghe: caricarle come file;
- datepicker: impostare prima la data di fine, poi quella di inizio;
- per filtri come `days = 91` usare il conditional builder;
- l'output in parquet a volte fallisce: usare csv.gz.

---

## 3. Codice nel repository (Python; numpy, pandas, scipy, matplotlib; niente statsmodels)

Esecuzione in ordine, con `RAW_DIR` = cartella dei dati grezzi e `DATA_DIR` = cartella di lavoro (i file intermedi sono pickle, perché l'ambiente originale non aveva un motore parquet; si può tornare a parquet):

```
python -m src.build_data RAW_DIR DATA_DIR        # members_daily, weights_monthly, spx_daily, iv_members, zero_curve
python -m src.validate_data RAW_DIR DATA_DIR     # report di qualità
python -m src.signals DATA_DIR                   # panel mensile: ic, rc21/63/252, rc_fwd (53 gg), iv_spx91, rv_spx21, vp, vix, coverage
python -m src.forecast DATA_DIR results          # HAR stimato sul 1996–2010 e congelato; aggiunge cp = ic − fcst
python -m src.simple_dispersion DATA_DIR results # parte 2: P&L esatto degli straddle delta-hedgiati
python -m src.simple_dispersion_report DATA_DIR RAW_DIR results   # statistiche, fattori, skew SPX, grafici
python -m src.premium DATA_DIR results           # parte 1 (bozza): premio a 1M e 3M, per periodo e per VIX
```

**Moduli:**
- `src/corr.py`: correlazione media implicita/realizzata, ρ = (σ_I² − Σwᵢ²σᵢ²) / ((Σwᵢσᵢ)² − Σwᵢ²σᵢ²), più Fisher z.
- `src/stats.py`: OLS con errori Newey-West, Sharpe, max drawdown, block bootstrap della differenza di Sharpe, deflated Sharpe.
- `src/build_data.py`, `src/signals.py`, `src/forecast.py`, `src/simple_dispersion.py`, `src/simple_dispersion_report.py`, `src/premium.py`, `src/validate_data.py`.

**Definizioni chiave, come nella pre-registrazione:**
- **Date d'ingresso:** primo giorno di borsa del mese.
- **Pesi:** capitalizzazione CRSP a fine mese precedente; secid valido alla data tramite il link.
- **IC:** IV ATM-forward a 91 giorni (media di call e put) di SPX e di **tutti** i membri. La data è scartata se la copertura del peso è sotto il 90%.
- **RC:** stessa formula con vol realizzate da rendimenti logaritmici. Un titolo è escluso se ha meno dell'80% di osservazioni valide.
- **Target del forecast:** RC sui 53 giorni di borsa successivi.
- **Forecast HAR:** z(RC_fwd) = a + b₁z(RC21) + b₂z(RC63) + b₃z(RC252) + b₄z(IC), con Newey-West a 3 lag. È stimato **solo su date il cui target finisce entro il 31/12/2010**: è più prudente della pre-registrazione e va dichiarato nel paper. Coefficienti congelati in `results/forecast_frozen.json`.

---

## 4. Risultati ottenuti finora

**Forecast** (stima con n = 166):
- coefficienti: rc21 0,23 (t 2,4), rc63 0,13 (t 1,0), rc252 0,31 (t 2,1), ic 0,04 (t 0,3);
- R² 0,42 su Fisher z;
- RMSE nel periodo di stima 0,088;
- RMSE fuori campione 2011–2025: HAR 0,134, solo RC21 0,177, solo RC63 0,159. Il forecast combinato batte le finestre singole.

**Parte 1, bozza non ancora committata su GitHub per decisione di Luca** (attendere tutti i dati prima di proseguire):

| Periodo | IC 3M | RC dopo, 3M | Premio 3M (t NW) | Premio 1M (t NW) |
|---|---|---|---|---|
| 1996–2007 | 0,375 | 0,271 | 0,104 (7,1) | 0,076 (7,1) |
| 2008–2009 | 0,492 | 0,429 | 0,063 (1,9) | 0,050 (2,0) |
| 2010–2019 | 0,430 | 0,343 | 0,086 (5,0) | 0,042 (3,1) |
| 2020–2025 | 0,367 | 0,319 | 0,048 (2,0) | 0,016 (1,0) |
| Totale | 0,400 | 0,315 | 0,085 (8,4) | 0,051 (6,8) |

- Per terzile di VIX, il premio a 3M vale 0,04 con VIX basso, 0,10 con VIX medio e 0,11 con VIX alto.
- Regredendo il premio a 3M sul livello di IC si ottiene un beta di 0,51 (t 6,5).

**Parte 2: dispersion trade standard** (`results/SIMPLE_DISPERSION.md`):
- **Costruzione del trade:** ogni mese short straddle SPX ATM-forward a 3 mesi, long straddle sui top-50, vega-neutral con vega allocata per peso, delta-hedge giornaliero all'IV d'ingresso, tenuta a scadenza (63 giorni).
- **Calcolo del P&L:** esatto sul percorso reale dei prezzi (CRSP: retx per il prezzo, ret per i dividendi; dividend yield trailing; forward SPX dalla stdopd), con financing alla curva zero. È espresso in punti di vol per unità di vega.
- **Costi ipotizzati:** half-spread di 0,15 punti sull'indice e 0,50 sui single stock, pagati all'ingresso.

| Periodo | Media netta (punti di vol per tranche) | t NW | Sharpe |
|---|---|---|---|
| 1996–2007 | 2,13 | 4,55 | 1,17 |
| 2008–2009 | 1,22 | 0,83 | 0,41 |
| 2010–2019 | 0,25 | 0,97 | 0,24 |
| 2020–2025 | 1,04 | 1,56 | 0,56 |
| Totale | 1,23 | 4,57 | 0,69 |

- Al lordo dei costi la media è 1,88 e lo Sharpe 1,05.
- Le due gambe: short SPX +2,14, long titoli −0,26.
- Sensitività ai costi: con costi 2× lo Sharpe è 0,32, con 3× è −0,04.
- Regressioni sui fattori: alpha circa 1,6 (t > 5) con CAPM, FF5 + momentum e ΔVIX; R² ≤ 0,20. Il beta di mercato è negativo, perché un book vega-neutral è lungo il livello di vol: per questo 2008 e 2020 hanno dato tranche positive.
- Rispetto al solo short straddle SPX: R² 0,01, alpha 1,33 (t 3,6).
- Strike model-free del variance swap SPX a 3 mesi: in media 20,3 contro 18,6 di IV ATM. Uno short var swap allo strike corretto rende +1,7 punti; allo strike ATM circa 0.
- L'approssimazione "vega × (RV − IV)" ha correlazione solo 0,64 con il P&L esatto.

**Controlli ulteriori:**
- La versione esatta della formula IC (senza espansione di Taylor, uguaglianza delle varianze risolta numericamente) differisce al massimo di 0,55 punti, quindi è trascurabile.
- La sottostima dovuta all'approssimazione lognormale (Linders e Schoutens, 2014) resta da quantificare. Richiede gli smile dei membri.

---

## 5. Cosa resta da fare, in ordine

1. **Estrazione delle opzioni listate** (passi A e B) e delle superfici di volatilità dei membri sul WRDS JupyterHub. Verificare i dati: spread, quote mancanti, salti di IV.
2. **Parte 2 con i costi reali.** Sostituire gli spread ipotizzati con quelli osservati anno per anno. Separare i periodi prima/dopo la decimalizzazione (2001) e il penny pilot (2007). È decisivo, perché gran parte del profitto storico sta nel 1996–2002, quando gli spread erano larghi.
3. **Parte 1 completa.**
   - Stesura e grafici.
   - Confronto con l'indice di correlazione implicita del Cboe (COR3M): attenzione, il Cboe usa i top-50 e non tutti i 500 titoli.
   - Replica di Marshall (2009) sul 2005–2007 come controllo: il paper trova circa 1,2 punti di vol di scarto con le call a 30 giorni.
   - Robustezza alla Linders-Schoutens.
4. **Proporre e pre-registrare la parte 4** (il modello complesso) prima di toccare il test della parte 3.
5. **Parte 3:**
   - costo *c* stimato sul 1996–2010 dagli spread dei contratti scelti;
   - motore di P&L sui contratti listati;
   - ladder fino a 3 tranche, ciascuna con 1/3 della vega;
   - uscita 10 giorni di borsa prima della scadenza, acquisti all'ask e vendite al bid;
   - delta hedge giornaliero con i delta IvyDB, costi di hedging 0,5 bp per l'indice e 1 bp per i titoli;
   - quote mancanti portate avanti al massimo 3 giorni, poi chiusura della gamba;
   - benchmark incondizionato riscalato alla stessa vega media;
   - le 8 varianti dichiarate;
   - kill test: Sharpe migliore del benchmark con p < 0,10 al block bootstrap (blocchi di 3 mesi, 10.000 estrazioni), drawdown ≤ 2/3 di quello del benchmark, robustezza per k ∈ {0; 0,25; 0,5; 0,75; 1}. Tutti i dettagli sono in `PREREGISTRATION.md`.
6. **Paper** nel formato scelto da Luca; aggiornare README e ROADMAP.
7. **Facoltativo:** appendice da settembre 2025 a oggi con dati Bloomberg (Luca può estrarli dal terminale Bocconi).

---

## 6. Letteratura già discussa con Luca

- **Driessen, Maenhout e Vilkov (2009, Journal of Finance):** prezzo del rischio di correlazione. È il riferimento principale.
- **Marshall (2009, Global Finance Journal):** IV dell'indice contro una IV "Markowitz" con correlazioni storiche, 2005–2007. Limiti:
  - non misura un P&L;
  - mescola misure implicite e storiche;
  - fa un bootstrap iid su una serie autocorrelata;
  - usa un costo grezzo di 4,09 punti di vol.
  È utile per ricordare che i costi pre-2001 erano alti.
- **Linders e Schoutens (2014, Journal of Computational and Applied Mathematics):** la formula tradizionale della correlazione implicita, che è la nostra, sottostima quella del modello in media di circa 1,5 punti percentuali all'ATM sul Dow Jones 2000–2009, di più nelle crisi e fuori dal denaro. All'ATM è giudicata adeguata. La nostra misura è quindi prudente, ma con un bias che dipende dal regime: va dichiarato.
