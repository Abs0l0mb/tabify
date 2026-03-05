'use strict';

import {
    Popup,
    Api,
    Form,
    FormField,
    TextInput,
    BigTextInput
} from '@src/classes';

export class EditCategoryPopup extends Popup {

    private form: Form;

    private title: FormField;
    private description: FormField;

    constructor(private id: number) {

        super({
            validText: 'Update',
            cancellable: true,
            title: `Edit category ${id}`,
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

        this.populate();
    }

    /*
    **
    **
    */
    private async populate() : Promise<void> {
        
        try {

            let data: any = await Api.get('/category', {
                id: this.id
            });

            this.title.input.setValue(data.title);
            this.description.input.setValue(data.description);

            this.ready();
            
        } catch(error: any) {

            console.log(error);
        }
    }

    /*
    **
    **
    */
    public async onValid() : Promise<void> {

        this.validButton.load();

        try {

            await Api.post('/category/update', {
                id: this.id,
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