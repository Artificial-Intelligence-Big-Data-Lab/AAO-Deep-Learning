# Convertire CSV to Mysql

Nel percorso `./src/utils/csv_to_mysql` si trova un utile script per convertire i csv in una tabella mysql. 

La configurazione per la connessione al DB si trova nello stesso file. 

E' un file sconnesso rispetto agli altri file del framework, in quanto si tratta soltanto un file di appoggio per convertire i CSV in una tabellla MySql in modo da poter centralizzare i dati sul server e poter essere riutilizzati da chiunque in futuro. 

Per creare una tabella si può utilizzare il file `./sql/dataset_general.sql`, cambiare in modo appropriato il nome della tabella ed eseguire da `PhpMyAdmin` o qualsiasi altro software simile come `Navicat` etc. 
Ogni tabella, rappresenterà quindi un dataset. 

Il file in questione è il seguente: 

```sql
SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for bund
-- ----------------------------
DROP TABLE IF EXISTS `NOME TABELLA`;
CREATE TABLE `NOME TABELLA`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `date` date NOT NULL,
  `time` time(0) NOT NULL,
  `open` float NOT NULL,
  `close` float NOT NULL,
  `high` float NOT NULL,
  `low` float NOT NULL,
  `up` int(11) NOT NULL,
  `down` int(11) NOT NULL,
  `volume` float NOT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1201668 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Compact;

SET FOREIGN_KEY_CHECKS = 1;

```

Per eseguire il file: 

```bash
python csv_to_mysql.py
```