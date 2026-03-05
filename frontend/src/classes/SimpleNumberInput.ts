'use strict';

import {
    Div,
    Block
} from '@src/classes';

export interface SimpleNumberInputData {
    label?: string;
    value?: number;
    class?: string;
}

export class SimpleNumberInput extends Div {

    protected input: Block;
    protected icon: Div;
    protected value: number;

    constructor(public data: SimpleNumberInputData, parent?: Block) {

        super('simple-number-input', parent);
        
        if (this.data.class)
            this.addClass(this.data.class);
        
        if (this.data.label)
            new Div('label', this).write(this.data.label);

        this.input = new Block('input', {
            type: 'number'
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

        this.emit('value', new Number(this.input.element.value).valueOf());
    }

    /*
    **
    **
    */
    public setValue(value: number, emit: boolean = true) : void {

        this.input.element.value = value;
        
        if (emit)
            this.emit('value', this.value);
    }

    /*
    **
    **
    */
    public getValue() : number | null {
        
        return this.value;
    }
}