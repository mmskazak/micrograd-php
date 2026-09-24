<?php

declare(strict_types=1);

require __DIR__ . '/src/Value.php';
require __DIR__ . '/src/Neuron.php';
require __DIR__ . '/src/Layer.php';
require __DIR__ . '/src/MLP.php';
require __DIR__ . '/src/Digits.php';
require __DIR__ . '/src/WeightStore.php';
require __DIR__ . '/src/Trainer.php';

const DB_PATH = __DIR__ . '/data/micrograd.sqlite';
const INPUTS = 9;          // 9 палочек шаблона индекса
const HIDDEN = [12, 8];    // два скрытых слоя
const OUTPUTS = 10;        // нейроны цифр 0..9
const DEFAULT_LR = 0.3;

/**
 * Модель с весами из SQLite. Если в базе ничего нет (или другая архитектура) —
 * создаётся новая случайная сеть и сразу сохраняется.
 *
 * @return array{0: MLP, 1: WeightStore}
 */
function loadModel(): array
{
    $store = new WeightStore(DB_PATH);
    $model = new MLP(INPUTS, [...HIDDEN, OUTPUTS]);
    if ($store->hasWeightsFor($model->sizes)) {
        $store->load($model);
    } else {
        $store->reset();
        $store->save($model);
    }
    return [$model, $store];
}
