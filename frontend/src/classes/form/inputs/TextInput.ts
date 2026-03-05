'use strict';

import {
    InputStructure,
    Block
} from '@src/classes';

export interface TextInputData {
    label: string;
    mandatory?: boolean;
    maxLength?: number;
    autocomplete?: boolean;
    class?: string;
}

export class TextInput extends InputStructure {

    protected input: Block;

    constructor(public data: TextInputData, parent?: Block) {

        super(data, parent);

        this.setCustomType('text');
        
        this.input = new Block('input', {
            type: 'text',
            autocomplete: this.data.autocomplete ? this.data.autocomplete : 'none',
            maxlength: this.data.maxLength ? this.data.maxLength : 'none',
            novalidate: true
        }, this.inputContainer);

        this.input.onNative('input', this.onInput.bind(this));
    }

    /*
    **
    **
    */
    private onInput(emit: boolean = true) : boolean {

        const isFilled = this.input.element.value.length > 0 
        || this.input.element.matches('*:autofill')
        || this.input.element.matches('*:-webkit-autofill');

        this.setFilled(isFilled);
        
        if (emit)
            this.emit('value', this.input.element.value);

        return isFilled;
    }

    /*
    **
    **
    */
    public setValue(value: string | number) : void {
        
        this.input.element.value = value;

        this.onInput(false);
    }

    /*
    **
    **
    */
    public getValue() : string | number {

        return this.input.element.value.toString();
    }

    /*
    **
    **
    */
    public detectAutoFill() : void {

        let i = 0;

        const interval = setInterval(() => {

            if (i === 20 || this.onInput(false)) {
                clearInterval(interval);
                return;
            }

            i++;

        }, 100);
    }
}