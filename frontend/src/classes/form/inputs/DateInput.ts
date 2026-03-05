'use strict';

import {
    InputStructure,
    Block,
    DateSelection
} from '@src/classes';

export interface DateInputData {
    label: string;
    value?: Date;
    mandatory?: boolean;
    class?: string;
}

export class DateInput extends InputStructure {

    protected input: Block;
    protected date: Date | null;

    constructor(public data: DateInputData, parent?: Block) {

        super(data, parent);

        this.setCustomType('date');
        
        this.input = new Block('input', {
            type: 'text'
        }, this.inputContainer);
        
        this.customTypeIcon.onNative('mousedown', this.displayDateSelection.bind(this));
        
        this.input.onNative('input', this.onInput.bind(this));

        if (this.data.value)
            this.setValue(this.data.value, false);
    }

    /*
    **
    **
    */
    private onInput() : void {

        this.setDateFromText(this.input.element.value);
        this.setFilled(this.input.element.value.length > 0);

        this.emit('value', this.date);
    }

    /*
    **
    **
    */
    public setValue(date: Date | null, emit: boolean = true) : void {

        this.input.element.value = date ? date.toLocaleDateString('fr') : '';
        
        this.date = date;

        this.setDateFromText(this.input.element.value);
        this.setFilled(this.input.element.value.length > 0);

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

        this.date = date;
    }

    /*
    **
    **
    */
    private displayDateSelection(event: MouseEvent) : void {
        
        this.setCustomTypeIconActivated(true);

        const options: any = {};

        if (this.date)
            options.date = this.date;

        const selection = new DateSelection(event.pageX, event.pageY, options);
        
        selection.on('date', this.setValue.bind(this));
        selection.on('hide', () => this.setCustomTypeIconActivated(false));
    }
}