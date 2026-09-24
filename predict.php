<?php

declare(strict_types=1);

// php predict.php 1,0,1,0,0,1,1,0,1   — 9 палочек (0/1) в порядке их номеров
require __DIR__ . '/bootstrap.php';

$x = array_map('intval', explode(',', $argv[1] ?? ''));
if (count($x) !== INPUTS) {
    fwrite(STDERR, "Нужно " . INPUTS . " чисел через запятую, например: php predict.php " . implode(',', Digits::toInput(0)) . "\n");
    exit(1);
}

[$model] = loadModel();
$outs = $model(Digits::encode($x));
foreach ($outs as $d => $v) {
    $bar = str_repeat('█', (int)round(($v->data + 1) * 10));
    printf("%d  %+.3f  %s\n", $d, $v->data, $bar);
}
echo "Ответ сети: " . Trainer::argmax($outs) . "\n";
