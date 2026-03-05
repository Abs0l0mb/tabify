'use strict';

import { 
    Block,
    Div,
    FormField,
    ApiErrors,
    Popup 
} from '@src/classes';

export class Form extends Block {

    public fields: FormField[] = [];

    constructor(parent?: Block) {
        
        super('form', {}, parent);
    }

    /*
    **
    **
    */
    public add(input: Block, parent: Block = this) : FormField {

        input.appendTo(parent);
        
        const messageContainer = new Div('field-message', this);
        const field = new FormField(input, messageContainer);

        this.fields.push(field);

        if (input.getData('custom-type') !== 'big-text') {
            input.onNative('keydown', (event: KeyboardEvent) => {
                if (event.key === 'Enter') {
                    (document.activeElement as HTMLElement).blur();
                    this.emit('enter-down');
                }
            });
        }

        return field;
    }

    /*
    **
    **
    */
    public displayError(error: any) : void {
        
        this.clearErrors();

        //==================
        //UNMANAGEABLE ERROR
        //==================

        if (typeof error !== 'string') {
            console.log('Unmanageable error', error);
            return;
        }

        //===========
        //FIELD ERROR
        //===========

        const split = error.split('@');
        const field = this.getFieldByErrorKey(split[0]);
        const fieldMessage = ApiErrors.getMessage(split[1]);

        if (field && fieldMessage)
            return field.displayError(fieldMessage);

        //===============
        //NON FIELD ERROR
        //===============

        let message = ApiErrors.getMessage(error);

        if (!message)
            message = error;
        
        return this.displayPopupError(message);
    }

    /*
    **
    **
    */
    public getFieldByErrorKey(key: string) : FormField | null {

        for (let field of this.fields)
            if (field.errorKeys.includes(key))
                return field;
        
        return null;
    }

    /*
    **
    **
    */
    private displayPopupError(message: string) : void {

        new Popup({
            title: 'Error received',
            class: 'smaller',
            message: message,
            validText: 'Ok',
            cancellable: false,
            ready: true
        });
    }

    /*
    **
    **
    */
    public clearErrors() : void {

        for (let field of this.fields)
            field.hideError();
    }

    /*
    **
    **
    */
    public clearMessages() : void {

        for (let field of this.fields)
            field.hideMessage();
    }
}