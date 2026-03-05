'use strict';

import {
    Block,
    Div
} from '@src/classes';

export interface CheckboxData {
    label: string;
    value?: any;
    class?: string;
    textBlock?: Block;
}

export class Checkbox extends Div {

    private checkbox: Div;
    private labelContainer: Div;
    private value: boolean = false;

    constructor(public data: CheckboxData, parent?: Block) {

        super('checkbox-container', parent);

        if (this.data.class)
            this.addClass(this.data.class);

        this.value = false;

        this.checkbox = new Div('checkbox', this);
        new Div('check', this.checkbox);

        this.labelContainer = new Div('label-container', this);
        
        this.checkbox.onNative('mousedown', this.toggleValue.bind(this));
        this.labelContainer.onNative('mousedown', this.toggleValue.bind(this));

        if (this.data.textBlock)
            this.labelContainer.append(this.data.textBlock);
        else if (this.data.label)
            this.labelContainer.write(this.data.label);

        if (this.data.value)
            this.setValue(this.data.value);
    }

    /*
    **
    **
    */
    private toggleValue() : void {

        this.value = !this.value; 
        
        this.setValue(this.value, true);
    }

    /*
    **
    **
    */
    public setValue(value: boolean, emit: boolean = false) : void {

        this.value = value;
        
        if (emit)
            this.emit('value', this.value);

        this.setData('checked', this.value ? 1 : 0);
    }

    /*
    **
    **
    */
    public getValue() : boolean {

        return this.value;
    }

    /*
    **
    **
    */
    public enable() : void {

        this.setData('enabled', 1);
    }

    /*
    **
    **
    */
    public disable() : void {
        
       this.setData('enabled', 0);
    }
}