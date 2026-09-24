<?php

declare(strict_types=1);

// php train.php [эпох=200] [lr=0.3] [--reset]
require __DIR__ . '/bootstrap.php';

$args = array_values(array_filter(array_slice($argv, 1), fn($a) => $a !== '--reset'));
$epochs = (int)($args[0] ?? 200);
$lr = (float)($args[1] ?? DEFAULT_LR);

if (in_array('--reset', $argv, true)) {
    (new WeightStore(DB_PATH))->reset();
    echo "Веса сброшены, начинаем со случайных.\n";
}

[$model, $store] = loadModel();

for ($i = 0; $i < $epochs; $i++) {
    ['loss' => $loss, 'accuracy' => $acc] = Trainer::epoch($model, $lr);
    $epoch = $store->logEpoch($loss, $acc, $lr);
    if ($i % 25 === 0 || $i === $epochs - 1) {
        printf("эпоха %4d   loss = %.5f   угадано %d/10\n", $epoch, $loss, (int)round($acc * 10));
    }
}

$store->save($model);
echo "Веса сохранены в " . DB_PATH . "\n";
