'use strict';

import {
    Div,
    Block,
    DateSelection
} from '@src/classes';

export interface SimpleDateInputData {
    label?: string;
    value?: string | Date | null | undefined;
    class?: string;
}

export class SimpleDateInput extends Div {

    protected input: Block;
    protected icon: Div;
    protected date: Date | null;

    constructor(public data: SimpleDateInputData, parent?: Block) {

        super('simple-date-input', parent);
        
        if (this.data.class)
            this.addClass(this.data.class);

        if (this.data.label)
            new Div('label', this).write(this.data.label);

        this.input = new Block('input', {
            type: 'text'
        }, this);

        this.input.onNative('input', this.onInput.bind(this));
        
        this.icon = new Div('icon', this).onNative('mousedown', this.displayDateSelection.bind(this));

        if (this.data.value)
            this.setValue(this.data.value, false);

        console.log(data);
    }

    /*
    **
    **
    */
    private onInput() : void {

        this.setDateFromText(this.input.element.value);
        
        this.emit('value', this.date);
    }

    /*
    **
    **
    */
    public setValue(date: string | Date | null | undefined, emit: boolean = true) : void {

        if (!date)
            return;
        
        if (typeof date === 'string')
            date = new Date(date);

        this.input.element.value = date ? date.toLocaleDateString('fr') : '';

        this.setDateFromText(this.input.element.value);
        
        if (emit)
            this.emit('value', this.date);
    }

    /*
    **
    **
    */
    public getValue() : Date | null {
        
        return this.date;
    }

    /*
    **
    **
    */
    private setDateFromText(text: string) : void {

        const split = text.split('/');

        const day = parseInt(split[0]);
        const month = parseInt(split[1]);
        const year = parseInt(split[2]);

        if (!day || !month || !year) {
            this.date = null;
            return;
        }
        
        const date = new Date(Date.UTC(year, month-1, day));

        if (date instanceof Date && isNaN(date.getTime())) {
            this.date = null;
            return;
        }

        date.setHours(8);

        this.date = date;
    }

    /*
    **
    **
    */
    private displayDateSelection(event: MouseEvent) : void {

        const options: any = {};

        if (this.date)
            options.date = this.date;

        const selection = new DateSelection(event.pageX, event.pageY, options);
        
        selection.on('date', this.setValue.bind(this));
    }
}