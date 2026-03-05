'use strict';

import { Div } from '@src/classes';

export class FormField {

    public errorKeys: string[] = [];

    constructor(public input: any, private messageContainer: Div) {
    }

    /*
    **
    **
    */
    public displayError(message: string) : void {
        
        this.displayMessage(message, 'error');
    }

    /*
    **
    **
    */
    public displayInformation(message: string) : void {
        
        this.displayMessage(message, 'information');
    }

    /*
    **
    **
    */
    private displayMessage(message: string, style: string) : void {

        this.messageContainer
            .html(message)
            .setData('style', style)
            .setData('displayed', 1);
    }

    /*
    **
    **
    */
    public hideError() : void {
        
        if (this.messageContainer.getData('style') === 'error')
            this.messageContainer.setData('displayed', 0);
    }

    /*
    **
    **
    */
    public hideMessage() : void {
        
        this.messageContainer.setData('displayed', 0);
    }

    /*
    **
    **
    */
    public linkToErrorKey(key: string) : void {
        
        this.errorKeys.push(key);
    }

    /*
    **
    **
    */
    public linkToErrorKeys(keys: string[]) : void {

        for (let key of keys)
            this.linkToErrorKey(key);
    }
}