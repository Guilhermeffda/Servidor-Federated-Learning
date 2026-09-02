# Teste Global — TACO-10

Este diretório contém o conjunto de teste global utilizado para avaliar os modelos treinados no projeto FedLitter.

## Divisão

- Imagens totais do TACO: 1500
- Imagens de teste: 300
- Proporção: 20%
- Seed: 42
- Instâncias no teste: 1068

O teste global foi separado antes da criação das partições dos clientes. Portanto, nenhuma imagem deste conjunto deve ser utilizada no treinamento ou validação de qualquer cliente.

## Estratificação

A divisão foi realizada com `train_test_split` do scikit-learn utilizando `random_state=42` e estratificação.

Como uma imagem pode conter várias categorias, foi definida uma categoria principal por imagem: a categoria com maior número de instâncias na imagem. Essa categoria foi utilizada apenas para orientar a estratificação.

## Distribuição das instâncias

| Categoria | Instâncias |
|---|---:|
| Can | 59 |
| Other | 389 |
| Bottle | 101 |
| Bottle cap | 65 |
| Cup | 43 |
| Lid | 25 |
| Plastic bag + wrapper | 159 |
| Pop tab | 24 |
| Straw | 55 |
| Cigarette | 148 |

## Controle de sobreposição

Os IDs das imagens selecionadas para o teste global são armazenados em `test_image_ids.txt`.

O particionamento dos clientes deverá consultar essa lista e verificar, por meio de `assert`, que nenhuma imagem do teste global foi atribuída a um cliente.

O objetivo é garantir que FedAvg, FedProx e FedTrimmed sejam avaliados exatamente sobre os mesmos exemplos, sem vazamento de dados entre treinamento e teste.
