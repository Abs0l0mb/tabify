'use strict';

import {
    Popup,
    Api,
    Form,
    FormField,
    TextInput,
    BigTextInput
} from '@src/classes';

export class AddCategoryPopup extends Popup {

    private form: Form;

    private title: FormField;
    private description: FormField;

    constructor(private teamId: number) {

        super({
            validText: 'Add',
            cancellable: true,
            title: `Add category`,
        });

        this.build();
    }

    /*
    **
    **
    */
    private async build() : Promise<void> {

        this.form = new Form(this.content);

        //=====
        //TITLE
        //=====

        this.title = this.form.add(new TextInput({
            label: 'Title',
            mandatory: true
        }));

        this.title.linkToErrorKey('title');

        //===========
        //DESCRIPTION
        //===========

        this.description = this.form.add(new BigTextInput({
            label: 'Description',
            mandatory: true
        }));

        this.description.linkToErrorKey('description');

        this.ready();
    }

    /*
    **
    **
    */
    public async onValid() : Promise<void> {

        this.validButton.load();

        try {

            await Api.post('/team/categories/add', {
                teamId: this.teamId,
                title: this.title.input.getValue(),
                description: this.description.input.getValue(),
            });

            this.emit('success');
            this.hide();
                        
        } catch(error: any) {

            this.form.displayError(error);
            this.validButton.unload();
        }
    }
}