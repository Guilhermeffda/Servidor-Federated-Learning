# Estratégia de data augmentation pós-Dirichlet

- SHA-256 do manifesto non-IID: `ca1520cd7f8257b6e1f1b1d571610ba6f2e112993b8bb6461adce03864f878d9`
- Ultralytics: `8.4.121`
- Selected alpha: `0.5`
- Fonte da análise de escassez: somente `train_full_instance_distribution`.
- Política: absent = 0; critical = 1–9; scarce = 10–15 instâncias.
- Pares absent/critical/scarce: 0/10/5
- Clientes envolvidos: client_0, client_1, client_2, client_4
- Classes envolvidas: Bottle, Bottle cap, Can, Cup, Lid, Pop tab, Straw

## Combinações sinalizadas

| Severity | Instâncias | Cliente | ID | Classe TACO-10 |
|---|---:|---:|---:|---|
| critical | 1 | client_2 | 7 | Pop tab |
| critical | 2 | client_4 | 7 | Pop tab |
| critical | 4 | client_2 | 0 | Can |
| critical | 4 | client_2 | 3 | Bottle cap |
| critical | 4 | client_4 | 0 | Can |
| critical | 6 | client_2 | 5 | Lid |
| critical | 7 | client_2 | 2 | Bottle |
| critical | 7 | client_2 | 8 | Straw |
| critical | 9 | client_1 | 5 | Lid |
| critical | 9 | client_4 | 5 | Lid |
| scarce | 10 | client_0 | 5 | Lid |
| scarce | 10 | client_0 | 7 | Pop tab |
| scarce | 10 | client_1 | 8 | Straw |
| scarce | 10 | client_2 | 4 | Cup |
| scarce | 12 | client_4 | 8 | Straw |

## Configuração candidata

```yaml
# Moderate HSV variation for plausible lighting and color changes.
hsv_h: 0.015
hsv_s: 0.50
hsv_v: 0.30
# Low rotation and translation preserve urban-scene geometry.
degrees: 5.0
translate: 0.05
# Conservative scale because many litter objects are small.
scale: 0.25
# Vertical flips are implausible for urban scenes; horizontal flips remain valid.
flipud: 0.0
fliplr: 0.50
# Moderate Mosaic trades visual diversity against apparent small-object size.
mosaic: 0.50
```

HSV moderado amplia variações plausíveis de iluminação e cor. A rotação e translação são baixas. A escala é conservadora devido aos objetos pequenos. Flip vertical fica desabilitado para cenas urbanas e o horizontal permanece permitido. Mosaic moderado equilibra diversidade visual com o risco de reduzir o tamanho aparente de objetos pequenos.

Os parâmetros foram aceitos pela configuração nativa do Ultralytics 8.4.121 e não são ajustados automaticamente.

## Previews nativos

Os previews usam `YOLODataset` com `augment=True`, a configuração candidata e `plot_images` para desenhar as bounding boxes originais e transformadas. Nenhum modelo, treino ou peso é utilizado.

- `augmentation_previews/client_0_image_1066.jpg`
- `augmentation_previews/client_0_image_363.jpg`
- `augmentation_previews/client_1_image_764.jpg`
- `augmentation_previews/client_2_image_614.jpg`
- `augmentation_previews/client_2_image_1089.jpg`
- `augmentation_previews/client_2_image_1145.jpg`
- `augmentation_previews/client_2_image_532.jpg`
- `augmentation_previews/client_4_image_1152.jpg`
- `augmentation_previews/client_4_image_433.jpg`

## Inspeção visual

Verificar se as boxes continuam alinhadas; objetos pequenos permanecem reconhecíveis; cortes não são excessivos; scale não reduz o lixo a tamanho impraticável; Mosaic preserva instâncias pequenas; cores e brilho permanecem plausíveis; flips são semanticamente plausíveis; e não há artefatos graves.

**visual_validation_status: pending_review**

## Limitações conhecidas

1. Augmentation não corrige ausência total de uma classe.
2. Augmentation não equivale à aquisição de novos exemplos reais.
3. A configuração é global ao pipeline de treino, não class-specific.
4. Classes extremamente raras continuam sujeitas a alta variância.
5. Mosaic pode reduzir o tamanho aparente de objetos pequenos e depende de validação visual.
6. As augmentations atuam nas imagens amostradas durante o treino, não alteram IDs e não aumentam por si sós a frequência de amostragem de classes.
7. Exemplos raros continuam com baixa diversidade semântica; a estratégia aumenta diversidade visual, mas não rebalanceia classes.
