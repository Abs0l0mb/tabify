'use strict';

import {
    TitledPage,
    Div,
    Button,
    FileInput,
    SimpleNumberInput,
    Form,
    Api
} from '@src/classes';

interface ParamDef {
    key: string;
    label: string;
    defaultValue: number;
}

export class TabifyPage extends TitledPage {

    private midiFileInput: FileInput;
    private submitButton: Button;
    private suggestButton: Button;
    private values: { [key: string]: number } = {};
    private inputs: { [key: string]: SimpleNumberInput } = {};

    constructor() {

        super('Tabify', 'tabify');
        this.build();
    }

    /*
    **
    **
    */
    private build(): void {

        const main = new Div('', this.content);
        main.setStyle('padding', '20px');

        // ── MIDI file ─────────────────────────────────────────────────
        const fileZone = new Div('light-zone', main);
        fileZone.setStyle('margin-bottom', '16px');
        const fileForm = new Form(fileZone);
        this.midiFileInput = new FileInput({ label: 'Sélectionner un fichier .mid' }, fileForm);

        const suggestRow = new Div('', fileZone);
        suggestRow.setStyles({ 'padding': '8px 16px' });
        this.suggestButton = new Button({ label: 'Suggest parameters' }, suggestRow);
        this.suggestButton.onNative('click', this.onSuggest.bind(this));

        // ── General ───────────────────────────────────────────────────
        this.buildGroup(main, 'Général', [
            { key: 'step',  label: 'Step (ticks)',    defaultValue: 60  },
            { key: 'gpq',   label: 'Quarter ticks',   defaultValue: 960 },
            { key: 'tempo', label: 'Tempo (BPM)',      defaultValue: 120 },
        ]);

        // ── Search ────────────────────────────────────────────────────
        this.buildGroup(main, 'Recherche', [
            { key: 'max_fret',     label: 'Max fret',     defaultValue: 20  },
            { key: 'per_pitch_k',  label: 'Per pitch K',  defaultValue: 4   },
            { key: 'chord_k',      label: 'Chord K',      defaultValue: 50  },
            { key: 'beam_size',    label: 'Beam size',    defaultValue: 100 },
        ]);

        // ── Local cost ────────────────────────────────────────────────
        this.buildGroup(main, 'Coût local', [
            { key: 'w_span',                label: 'w_span',               defaultValue: 1.0  },
            { key: 'w_high',                label: 'w_high',               defaultValue: 0.2  },
            { key: 'high_fret_threshold',   label: 'High fret threshold',  defaultValue: 19   },
            { key: 'w_open_bonus',          label: 'w_open_bonus',         defaultValue: 0    },
            { key: 'w_string_range',        label: 'w_string_range',       defaultValue: 0.15 },
            { key: 'preferred_min_fret',    label: 'Preferred min fret',   defaultValue: 5    },
            { key: 'preferred_max_fret',    label: 'Preferred max fret',   defaultValue: 17   },
            { key: 'w_preferred_zone',      label: 'w_preferred_zone',     defaultValue: -1.5 },
            { key: 'high_string_threshold', label: 'High string threshold',defaultValue: 2    },
            { key: 'w_high_string',         label: 'w_high_string',        defaultValue: 2.0  },
        ]);

        // ── String discontinuity ──────────────────────────────────────
        this.buildGroup(main, 'Discontinuité de cordes', [
            { key: 'w_holes',  label: 'w_holes',  defaultValue: 4   },
            { key: 'w_gap',    label: 'w_gap',    defaultValue: 0.6 },
            { key: 'w_blocks', label: 'w_blocks', defaultValue: 4   },
        ]);

        // ── Transition cost ───────────────────────────────────────────
        this.buildGroup(main, 'Coût de transition', [
            { key: 'w_jump',                 label: 'w_jump',                 defaultValue: 0.8  },
            { key: 'jump_power',             label: 'jump_power',             defaultValue: 1.2  },
            { key: 'jump_threshold',         label: 'jump_threshold',         defaultValue: 5    },
            { key: 'jump_threshold_penalty', label: 'jump_threshold_penalty', defaultValue: 3.0  },
            { key: 'w_avg_jump',             label: 'w_avg_jump',             defaultValue: 0.6  },
            { key: 'avg_jump_power',         label: 'avg_jump_power',         defaultValue: 1.3  },
            { key: 'w_span_change',          label: 'w_span_change',          defaultValue: 0.25 },
            { key: 'w_string_center',        label: 'w_string_center',        defaultValue: 3    },
            { key: 'close_jump_threshold',   label: 'close_jump_threshold',   defaultValue: 4.0  },
            { key: 'close_jump_bonus',       label: 'close_jump_bonus',       defaultValue: -1.2 },
            { key: 'rest_enter_penalty',     label: 'rest_enter_penalty',     defaultValue: 0.0  },
            { key: 'rest_exit_penalty',      label: 'rest_exit_penalty',      defaultValue: 0.0  },
            { key: 'w_streak',               label: 'w_streak',               defaultValue: 4.0  },
            { key: 'streak_min_len',         label: 'streak_min_len',         defaultValue: 4    },
            { key: 'streak_speed_threshold', label: 'streak_speed_threshold', defaultValue: 480  },
        ]);

        // ── Submit ────────────────────────────────────────────────────
        const footer = new Div('', main);
        footer.setStyle('margin-top', '8px');
        this.submitButton = new Button({ label: 'Générer la tablature' }, footer);
        this.submitButton.onNative('click', this.onSubmit.bind(this));
    }

    /*
    **
    **
    */
    private buildGroup(parent: Div, title: string, params: ParamDef[]): void {

        const section = new Div('light-zone', parent);
        section.setStyle('margin-bottom', '16px');

        const titleDiv = new Div('', section);
        titleDiv.setStyles({
            'padding':       '10px 16px',
            'font-size':     '13px',
            'font-weight':   '600',
            'color':         'rgba(0,0,0,0.5)',
            'border-bottom': '1px solid rgba(0,0,0,0.08)',
            'text-transform':'uppercase',
            'letter-spacing':'0.04em'
        }).write(title);

        const grid = new Div('', section);
        grid.setStyles({
            'display':   'flex',
            'flex-wrap': 'wrap',
            'gap':       '6px',
            'padding':   '12px 16px'
        });

        for (const { key, label, defaultValue } of params) {

            this.values[key] = defaultValue;

            const input = new SimpleNumberInput({ label, value: defaultValue }, grid);
            input.setStyles({
                'border':        '1px solid rgba(0,0,0,0.1)',
                'border-radius': '5px',
                'background':    '#e9e9e9'
            });

            input.on('value', (v: number) => {
                this.values[key] = v;
            });

            this.inputs[key] = input;
        }
    }

    /*
    **
    **
    */
    private async onSuggest(): Promise<void> {

        const { base64, name } = this.getMidiFile();

        if (!base64 || !name) {
            alert('Veuillez sélectionner un fichier MIDI.');
            return;
        }

        this.suggestButton.load();

        try {
            const suggested: { [key: string]: number } = await Api.post('/suggest-params', {
                midi_base64: base64,
                midi_name:   name,
            });

            for (const [key, value] of Object.entries(suggested)) {
                if (key in this.inputs) {
                    this.inputs[key].setValue(value, false);
                    this.values[key] = value;
                }
            }
        } catch (error) {
            alert(`Erreur : ${error}`);
        } finally {
            this.suggestButton.unload();
        }
    }

    /*
    **
    **
    */
    private async onSubmit(): Promise<void> {

        const { base64, name } = this.getMidiFile();

        if (!base64 || !name) {
            alert('Veuillez sélectionner un fichier MIDI.');
            return;
        }

        this.submitButton.load();

        try {
            const arrayBuffer: ArrayBuffer = await Api.request('POST', '/tabify', {
                midi_base64: base64,
                midi_name:   name,
                ...this.getValues()
            }, true);

            const blob = new Blob([arrayBuffer], { type: 'application/octet-stream' });
            const url  = URL.createObjectURL(blob);
            const a    = document.createElement('a');
            a.href     = url;
            a.download = name.replace(/\.midi?$/i, '') + '.gp5';
            a.click();
            URL.revokeObjectURL(url);

        } catch (error) {
            alert(`Erreur : ${error}`);
        } finally {
            this.submitButton.unload();
        }
    }

    /*
    **
    **
    */
    public getValues(): { [key: string]: number } {

        return { ...this.values };
    }

    /*
    **
    **
    */
    public getMidiFile(): { base64: string | null; name: string | null } {

        return {
            base64: this.midiFileInput.getValue(),
            name:   this.midiFileInput.getName(),
        };
    }
}
