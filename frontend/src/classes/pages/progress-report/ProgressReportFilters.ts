'use strict';

import {
    Block,
    Div,
    Form,
    SimpleDateInput,
    Button,
    Tools
} from '@src/classes';

export class ProgressReportFilters extends Div {

    private filtersContainer: Div;
    private startInput: SimpleDateInput;
    private endInput: SimpleDateInput;
    
    //private resetButton: Button;
    private generateButton: Button;

    private startDate: Date;
    private endDate: Date;
    
    constructor(parent: Block) {

        super('progress-report-filters', parent);
        
        this.startDate = this.getDefaultStartDate();
        this.endDate = this.getDefaultEndDate();

        this.drawFilters();
    }

    /*
    **
    **
    */
    private async drawFilters() : Promise<void> {

        if (!this.filtersContainer)
            this.filtersContainer = new Div('filters-container', this);

        const form = new Form(this.filtersContainer);

        //===========
        //START INPUT
        //===========

        this.startInput = new SimpleDateInput({
            label: 'Start date',
            value: this.startDate
        }, form);
        
        this.startInput.on('value', this.filter.bind(this));

        //=========
        //END INPUT
        //=========

        this.endInput = new SimpleDateInput({
            label: 'End date',
            value: this.endDate
        }, form);
        
        this.endInput.on('value', this.filter.bind(this));

        //============
        //RESET BUTTON
        //============

        /*this.resetButton = new Button({
            label: 'Reset'
        }, form).onNative('click', this.resetFilters.bind(this));*/

        //===============
        //GENERATE BUTTON
        //===============

        this.generateButton = new Button({
            label: 'Download docx'
        }, form).onNative('click', this.onGenerateButton.bind(this));

        await Tools.sleep(10);

        //this.computeResetButtonState();
        
        this.filtersContainer.setData('displayed', 1);
    }
    
    /*
    **
    **
    */
    private filter() {

        this.startDate = this.startInput.getValue() ? this.startInput.getValue()! : this.startDate;
        this.endDate = this.endInput.getValue() ? this.endInput.getValue()! : this.endDate;
        
        this.emit('value', {
            startDate: this.startDate,
            endDate: this.endDate
        });

        //this.computeResetButtonState();
    }

    /*
    **
    **
    */
    /*private resetFilters() : void {

        this.startInput.setValue(this.getDefaultStartDate(), false);
        this.endInput.setValue(this.getDefaultEndDate(), false);

        this.computeResetButtonState();
        this.filter();
    }*/

    /*
    **
    **
    */
    /*private computeResetButtonState() : void  {

        if (this.startInput.getValue() && this.endInput.getValue() && +this.startInput.getValue()?! === +this.getDefaultStartDate() && +this.endInput.getValue()! === +this.getDefaultEndDate())
            this.resetButton.disable();
        else
            this.resetButton.enable();
    }*/

    /*
    **
    **
    */
    private getDefaultStartDate() : Date {

        const now = new Date();

        return new Date(now.getFullYear(), now.getMonth(), 1);
    }

    /*
    **
    **
    */
    private getDefaultEndDate() : Date {

        return new Date(new Date().getFullYear(), new Date().getMonth() + 1, 0);
    }

    /*
    **
    **
    */
    public getStartDate() : Date {

        return this.startDate;
    }

    /*
    **
    **
    */
    public getEndDate() : Date {

        return this.endDate;
    }

    /*
    **
    **
    */
    private onGenerateButton() : void {

        this.emit('generate');
    }
}