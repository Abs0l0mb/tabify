'use strict';

import {
    Div,
    Block
} from '@src/classes';

export interface SimpleTextInputData {
    label: string;
    value?: string;
    class?: string;
}

export class SimpleTextInput extends Div {

    protected input: Block;
    protected icon: Div;

    constructor(public data: SimpleTextInputData, parent?: Block) {

        super('simple-text-input', parent);
        
        if (this.data.class)
            this.addClass(this.data.class);

        if (this.data.label)
            new Div('label', this).write(this.data.label);

        this.input = new Block('input', {
            type: 'text'
        }, this);

        this.input.onNative('input', this.onInput.bind(this));

        this.icon = new Div('icon', this);

        if (this.data.value)
            this.setValue(this.data.value, false);
    }

    /*
    **
    **
    */
    private onInput() : void {

        this.emit('value', this.input.element.value);
    }

    /*
    **
    **
    */
    public setValue(value: string | number, emit: boolean = true) : void {

        this.input.element.value = value.toString();
        
        if (emit)
            this.emit('value', this.input.element.value);
    }

    /*
    **
    **
    */
    public getValue() : string {
        
        return this.input.element.value;
    }
}