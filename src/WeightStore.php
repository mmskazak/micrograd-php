<?php

declare(strict_types=1);

/**
 * Хранение весов и истории обучения в SQLite.
 *
 * params:       одна строка = один вес или bias: (слой, нейрон, kind 'w'|'b', номер входа, значение)
 * training_log: loss и точность после каждой эпохи
 * meta:         архитектура сети и счётчик эпох
 */
final class WeightStore
{
    private PDO $pdo;

    public function __construct(string $path)
    {
        $this->pdo = new PDO('sqlite:' . $path);
        $this->pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
        $this->pdo->exec('
            CREATE TABLE IF NOT EXISTS params (
                layer  INTEGER NOT NULL,
                neuron INTEGER NOT NULL,
                kind   TEXT    NOT NULL CHECK (kind IN (\'w\', \'b\')),
                idx    INTEGER NOT NULL,
                value  REAL    NOT NULL,
                PRIMARY KEY (layer, neuron, kind, idx)
            );
            CREATE TABLE IF NOT EXISTS training_log (
                epoch    INTEGER PRIMARY KEY,
                loss     REAL NOT NULL,
                accuracy REAL NOT NULL,
                lr       REAL NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime(\'now\'))
            );
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        ');
    }

    /** Есть ли в базе веса именно для такой архитектуры */
    public function hasWeightsFor(array $sizes): bool
    {
        return $this->getMeta('sizes') === json_encode($sizes)
            && (int)$this->pdo->query('SELECT COUNT(*) FROM params')->fetchColumn() > 0;
    }

    public function save(MLP $model): void
    {
        $this->pdo->beginTransaction();
        $this->pdo->exec('DELETE FROM params');
        $stmt = $this->pdo->prepare('INSERT INTO params (layer, neuron, kind, idx, value) VALUES (?, ?, ?, ?, ?)');
        foreach ($model->layers as $l => $layer) {
            foreach ($layer->neurons as $n => $neuron) {
                foreach ($neuron->w as $i => $w) {
                    $stmt->execute([$l, $n, 'w', $i, $w->data]);
                }
                $stmt->execute([$l, $n, 'b', 0, $neuron->b->data]);
            }
        }
        $this->setMeta('sizes', json_encode($model->sizes));
        $this->pdo->commit();
    }

    /** Записать значения из базы в data параметров уже созданной модели */
    public function load(MLP $model): void
    {
        $rows = $this->pdo->query('SELECT layer, neuron, kind, idx, value FROM params');
        foreach ($rows as $r) {
            $neuron = $model->layers[(int)$r['layer']]->neurons[(int)$r['neuron']];
            if ($r['kind'] === 'w') {
                $neuron->w[(int)$r['idx']]->data = (float)$r['value'];
            } else {
                $neuron->b->data = (float)$r['value'];
            }
        }
    }

    public function logEpoch(float $loss, float $accuracy, float $lr): int
    {
        $epoch = $this->epoch() + 1;
        $this->pdo->prepare('INSERT INTO training_log (epoch, loss, accuracy, lr) VALUES (?, ?, ?, ?)')
            ->execute([$epoch, $loss, $accuracy, $lr]);
        $this->setMeta('epoch', (string)$epoch);
        return $epoch;
    }

    public function epoch(): int
    {
        return (int)($this->getMeta('epoch') ?? 0);
    }

    public function history(): array
    {
        return $this->pdo->query('SELECT epoch, loss, accuracy FROM training_log ORDER BY epoch')
            ->fetchAll(PDO::FETCH_ASSOC);
    }

    /** Стереть всё: веса, историю, счётчик эпох */
    public function reset(): void
    {
        $this->pdo->exec('DELETE FROM params; DELETE FROM training_log; DELETE FROM meta;');
    }

    private function getMeta(string $key): ?string
    {
        $stmt = $this->pdo->prepare('SELECT value FROM meta WHERE key = ?');
        $stmt->execute([$key]);
        $v = $stmt->fetchColumn();
        return $v === false ? null : $v;
    }

    private function setMeta(string $key, string $value): void
    {
        $this->pdo->prepare('INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)')->execute([$key, $value]);
    }
}
