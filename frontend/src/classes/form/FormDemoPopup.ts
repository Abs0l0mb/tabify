'use strict';

import {
    Popup,
    Form,
    FormField,
    TextInput,
    PasswordInput,
    NumberInput,
    PercentInput,
    SelectInput,
    AutocompleteInput,
    DateInput,
    InlineInputsContainer,
    Checkbox,
    Tools
} from '@src/classes';

export class FormDemoPopup extends Popup {

    private form: Form;

    private text: FormField;
    private password: FormField;
    private number: FormField;
    private percent: FormField;
    private select: FormField;
    private autocomplete: FormField;
    private date: FormField;
    private start: FormField;
    private end: FormField;
    private checkbox: FormField;

    constructor() {

        super({
            validText: 'Add',
            cancellable: true,
            title: `Add user`,
        });

        this.build();
    }

    /*
    **
    **
    */
    private async build() : Promise<void> {

        this.form = new Form(this.content);

        //====
        //TEXT
        //====

        this.text = this.form.add(new TextInput({
            label: 'Text',
            mandatory: true
        }));


        //========
        //PASSWORD
        //========

        this.password = this.form.add(new PasswordInput({
            label: 'Password',
            mandatory: true
        }));

        //======
        //NUMBER
        //======

        this.number = this.form.add(new NumberInput({
            label: 'Number',
            mandatory: true
        }));

        //=======
        //PERCENT
        //=======

        this.percent = this.form.add(new PercentInput({
            label: 'Percent',
            mandatory: true
        }));

        //======
        //SELECT
        //======

        this.select = this.form.add(new SelectInput({
            label: 'Select',
            mandatory: true,
            items: [
                { label: 'Test 1', value: 1 },
                { label: 'Test 2', value: 2 },
                { label: 'Test 3', value: 3 },
                { label: 'Test 4', value: 4 }
            ]
        }));

        //============
        //AUTOCOMPLETE
        //============

        this.autocomplete = this.form.add(new AutocompleteInput({
            label: 'Autocomplete',
            mandatory: true,
            itemsEndpoint: '/wbss/for-assignment',
            itemEndpoint: '/wbs',
            getItemLabel(data) { return `${data.SPPDTTSAP_LONGREFE} | ${data.SPPDTTSAP_DESC}` },
            getItemValue(data) { return data.SPPDTTSAP_ID },
            getInputText(data) { return data.SPPDTTSAP_LONGREFE }
        }));

        //========
        //CHECKBOX
        //========

        this.checkbox = this.form.add(new Checkbox({
            label: 'Checkbox'
        }));

        //====
        //DATE
        //====

        this.date = this.form.add(new DateInput({
            label: 'Date',
            mandatory: true
        }));

        //======
        //PERIOD
        //======

        const inline = new InlineInputsContainer(this.form);

        this.start = this.form.add(new DateInput({
            label: 'Start',
            mandatory: true
        }), inline);

        this.end = this.form.add(new DateInput({
            label: 'End',
            mandatory: true
        }), inline);

        //============
        //AUTOCOMPLETE
        //============

        this.autocomplete = this.form.add(new AutocompleteInput({
            label: 'Autocomplete',
            mandatory: true,
            itemsEndpoint: '/wbss/for-assignment',
            itemEndpoint: '/wbs',
            getItemLabel(data) { return `${data.SPPDTTSAP_LONGREFE} | ${data.SPPDTTSAP_DESC}` },
            getItemValue(data) { return data.SPPDTTSAP_ID },
            getInputText(data) { return data.SPPDTTSAP_LONGREFE }
        }));

        //await Tools.sleep(2000);
        this.ready();
    }
}