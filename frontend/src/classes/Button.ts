'use strict';

import {
    Block,
    Div
} from '@src/classes';

export interface ButtonData {
    label: string;
    class?: string;
}

export class Button extends Div {

    constructor(public data: ButtonData, parent?: Block) {

        super('button-container', parent);

        if (this.data.class)
            this.addClass(this.data.class);
            
        new Block('button', {
            type: 'button'
        }, this).write(this.data.label);

        new Div('spinner', this);
    }

    /*
    **
    **
    */
    public load() : void {

        this.setData('loading', 1);
    }

    /*
    **
    **
    */
    public unload() : void {

        this.setData('loading', 0);
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