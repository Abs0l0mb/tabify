'use strict';

import {
    InputStructure,
    Block
} from '@src/classes';

export interface BigTextInputData {
    label: string;
    mandatory?: boolean;
    class?: string;
}

export class BigTextInput extends InputStructure {

    protected input: Block;

    constructor(public data: BigTextInputData, parent?: Block) {

        super(data, parent);

        this.setCustomType('big-text');
        
        this.input = new Block('textarea', {}, this.inputContainer);

        this.input.onNative('input', this.onInput.bind(this));
    }

    /*
    **
    **
    */
    private onInput(emit: boolean = true) : boolean {

        const isFilled = this.input.element.value.length > 0;

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
}