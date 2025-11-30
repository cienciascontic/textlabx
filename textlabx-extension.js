(function (Scratch) {
  'use strict';

  const { ArgumentType, BlockType } = Scratch;

  // 👉 Cambiá esto cuando lo subas a Railway
  const BASE_URL = 'http://127.0.0.1:8000';

  class TextLabXExtension {
    constructor() {
      this.modelId = ''; // se setea desde un bloque
    }

    getInfo() {
      return {
        id: 'textlabx',
        name: 'TextLabX',
        color1: '#6a11cb',
        color2: '#2575fc',
        color3: '#4b2a99',
        blocks: [
          {
            opcode: 'setModel',
            blockType: BlockType.COMMAND,
            text: 'usar modelo [ID]',
            arguments: {
              ID: {
                type: ArgumentType.STRING,
                defaultValue: 'abc123'
              }
            }
          },
          {
            opcode: 'classify',
            blockType: BlockType.REPORTER,
            text: 'categoría de [TEXTO]',
            arguments: {
              TEXTO: {
                type: ArgumentType.STRING,
                defaultValue: 'Esta actividad está buenísima'
              }
            }
          }
        ]
      };
    }

    // BLOQUE: usar modelo [ID]
    setModel(args) {
      this.modelId = (args.ID || '').trim();
    }

    // BLOQUE: categoría de [TEXTO]
    async classify(args) {
      const texto = (args.TEXTO || '').trim();

      if (!texto) return '';
      if (!this.modelId) {
        return 'SIN MODELO';
      }

      try {
        const res = await fetch(`${BASE_URL}/predict/${this.modelId}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ texto })
        });

        if (!res.ok) {
          return 'ERROR ' + res.status;
        }

        const data = await res.json();
        return data.categoria || 'SIN RESPUESTA';
      } catch (e) {
        return 'ERROR FETCH';
      }
    }
  }

  Scratch.extensions.register(new TextLabXExtension());
})(Scratch);