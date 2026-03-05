'use strict';

import {
    Popup,
    Api,
    Form,
    FormField,
    TextInput,
    ClientLocation
} from '@src/classes';

export class AddTeamPopup extends Popup {

    private form: Form;

    private name: FormField;

    constructor() {

        super({
            validText: 'Add',
            cancellable: true,
            title: `Add team`,
        });

        this.build();
    }

    /*
    **
    **
    */
    private async build() : Promise<void> {

        this.form = new Form(this.content);

        //====
        //NAME
        //====

        this.name = this.form.add(new TextInput({
            label: 'Name',
            mandatory: true
        }));

        this.name.linkToErrorKey('name');

        this.ready();
    }

    /*
    **
    **
    */
    public async onValid() : Promise<void> {

        this.validButton.load();

        try {

            await Api.post('/teams/add', {
                name: this.name.input.getValue(),
            });

            this.emit('success');
            this.hide();
            
            ClientLocation.get().navigation.refresh();
            
        } catch(error: any) {

            this.form.displayError(error);
            this.validButton.unload();
        }
    }
}