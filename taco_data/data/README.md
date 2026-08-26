# Aquisição do dataset TACO

Esta pasta documenta a aquisição reproduzível do **TACO (Trash Annotations
in Context)**. O [repositório oficial dos
autores](https://github.com/pedropro/TACO) permanece como fonte documental do
dataset.

O artefato bruto primário adotado nesta pesquisa é o arquivo oficial
`TACO.zip`, publicado no Zenodo:

- Record: `3587843`
- DOI: `10.5281/zenodo.3587843`
- MD5 esperado: `e9149407d883e21a8d224feef8210920`

O script `scripts/download_dataset.py` armazena o archive em
`data/raw/taco/archives/` e preserva a estrutura interna real do ZIP ao
extraí-lo em `data/raw/taco/extracted/`. Todo o diretório `data/raw/taco/`
contém dados brutos ou metadados locais de aquisição, é ignorado pelo Git e
não deve ser versionado.

As estatísticas reais serão preenchidas somente depois da auditoria do
conteúdo extraído. 

**TODO:** registrar as estatísticas verificadas na etapa de
inspeção do dataset. 

