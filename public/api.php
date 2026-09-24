<?php

declare(strict_types=1);

/**
 * JSON API для визуализации.
 *
 * GET  ?action=state                         — архитектура, шаблон, веса, история обучения
 * POST ?action=forward {x, target}           — прямой + обратный проход для одного входа
 * POST ?action=train   {epochs, lr, detail}  — обучение, веса сохраняются в SQLite
 * POST ?action=reset                         — новые случайные веса, история стирается
 */

require __DIR__ . '/../bootstrap.php';

header('Content-Type: application/json; charset=utf-8');

function params(MLP $model): array
{
    $layers = [];
    foreach ($model->layers as $layer) {
        $neurons = [];
        foreach ($layer->neurons as $n) {
            $neurons[] = [
                'w' => array_map(fn(Value $v) => $v->data, $n->w),
                'b' => $n->b->data,
            ];
        }
        $layers[] = $neurons;
    }
    return $layers;
}

function grads(MLP $model): array
{
    $layers = [];
    foreach ($model->layers as $layer) {
        $neurons = [];
        foreach ($layer->neurons as $n) {
            $neurons[] = [
                'w' => array_map(fn(Value $v) => $v->grad, $n->w),
                'b' => $n->b->grad,
            ];
        }
        $layers[] = $neurons;
    }
    return $layers;
}

function node(Value $v): array
{
    return ['label' => $v->label, 'op' => $v->op, 'data' => $v->data, 'grad' => $v->grad];
}

/** Все узлы Value внутри нейрона после forward+backward — это и есть граф micrograd */
function trace(Neuron $n): array
{
    return [
        'x' => array_map('node', $n->trace['x']),
        'w' => array_map('node', $n->w),
        'b' => node($n->b),
        'muls' => array_map('node', $n->trace['muls']),
        'sums' => array_map('node', $n->trace['sums']),
        'out' => node($n->trace['out']),
    ];
}

function body(): array
{
    $json = json_decode(file_get_contents('php://input') ?: '{}', true);
    return is_array($json) ? $json : [];
}

try {
    $action = $_GET['action'] ?? 'state';
    [$model, $store] = loadModel();

    switch ($action) {
        case 'state':
            $digits = [];
            foreach (Digits::DIGITS as $d => $_) {
                $digits[$d] = Digits::toInput($d);
            }
            $out = [
                'sizes' => $model->sizes,
                'segments' => Digits::SEGMENTS,
                'digits' => $digits,
                'params' => params($model),
                'epoch' => $store->epoch(),
                'history' => $store->history(),
                'lr' => DEFAULT_LR,
            ];
            break;

        case 'forward':
            $in = body();
            $segments = array_map('intval', $in['x'] ?? Digits::toInput(0));
            $matched = Digits::match($segments);
            $target = isset($in['target']) ? (int)$in['target'] : $matched;

            // прямой проход
            $outs = $model(Digits::encode($segments));
            // незнакомая комбинация палочек и ответ не задан — считаем ответ сети "правильным"
            $target ??= Trainer::argmax($outs);

            // loss только для этого примера: Σ (ypred_k − y_k)²
            $ys = Digits::target($target);
            $loss = new Value(0.0);
            foreach ($outs as $k => $yk) {
                $diff = $yk->sub($ys[$k]);
                $loss = $loss->add($diff->mul($diff));
            }
            foreach ($model->parameters() as $p) {
                $p->grad = 0.0;
            }
            $loss->backward();

            $neurons = [];
            foreach ($model->layers as $layer) {
                $neurons[] = array_map('trace', $layer->neurons);
            }
            $out = [
                'x' => $segments,
                'encoded' => Digits::encode($segments),
                'matched' => $matched,
                'target' => $target,
                'prediction' => Trainer::argmax($outs),
                'outputs' => array_map(fn(Value $v) => $v->data, $outs),
                'loss' => $loss->data,
                'neurons' => $neurons,
                'grads' => grads($model),
            ];
            break;

        case 'train':
            $in = body();
            $epochs = max(1, min(500, (int)($in['epochs'] ?? 1)));
            $lr = (float)($in['lr'] ?? DEFAULT_LR);
            $before = params($model);
            $log = [];
            $perDigit = [];
            for ($i = 0; $i < $epochs; $i++) {
                ['loss' => $loss, 'accuracy' => $acc, 'perDigit' => $perDigit] = Trainer::epoch($model, $lr);
                $log[] = ['epoch' => $store->logEpoch($loss, $acc, $lr), 'loss' => $loss, 'accuracy' => $acc];
            }
            $store->save($model);
            $out = [
                'log' => $log,
                'epoch' => $store->epoch(),
                'before' => $before,
                'grads' => grads($model), // градиенты последней эпохи (по всем 10 цифрам)
                'perDigit' => $perDigit,  // Σ (ypred − y)² каждой цифры в последней эпохе, до update
                'params' => params($model),
            ];
            break;

        case 'reset':
            $store->reset();
            [$model, $store] = loadModel();
            $out = ['params' => params($model), 'epoch' => 0, 'history' => []];
            break;

        default:
            http_response_code(400);
            $out = ['error' => "Неизвестное действие: $action"];
    }

    echo json_encode($out, JSON_UNESCAPED_UNICODE | JSON_PRESERVE_ZERO_FRACTION);
} catch (Throwable $e) {
    http_response_code(500);
    echo json_encode(['error' => $e->getMessage()], JSON_UNESCAPED_UNICODE);
}
