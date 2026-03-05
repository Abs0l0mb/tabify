'use strict';

import {
    Block,
    Div
} from '@src/classes';

export interface InputStructureData {
    label: string;
    maxLength?: number;
    autocomplete?: boolean;
    mandatory?: boolean;
    class?: string;
}

export abstract class InputStructure extends Div {

    protected label: Div;
    protected inputContainer: Div;
    protected customTypeIcon: Div;

    constructor(public data: InputStructureData, parent?: Block) {

        super('input-structure', parent);

        this.label = new Div('label', this).write(this.data.label);
        this.inputContainer = new Div('input-container', this);
        this.customTypeIcon = new Div('custom-type-icon', this);
        
        if (this.data.mandatory)
            this.setData('mandatory', 1);
        
        if (this.data.class)
            this.addClass(this.data.class);
    }

    /*
    **
    **
    */
    protected setFilled(status: boolean) : void {
        
        this.setData('filled', status ? 1 : 0);
    }

    /*
    **
    **
    */
    protected setCustomType(value: string) : void {

        this.setData('custom-type', value);
    }
    
    /*
    **
    **
    */
    public abstract setValue(value: any) : void;

    /*
    **
    **
    */
    public abstract getValue() : any;

    /*
    **
    **
    */
    public setMandatory() : void {
        
        this.setData('mandatory', 1);
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

    /*
    **
    **
    */
    public setCustomTypeIconActivated(status: boolean) : void {
        
        this.setData('custom-type-icon-activated', status ? 1 : 0);
    }
}